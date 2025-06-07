from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from torch.nn import functional as F

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class GatingPAdapterPATASR(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        trans_type: str = "warp-rnnt",
        loss_rescaler_type: str = None,
        rescaler_model_conf: dict = None,
        simi_max_value: float = 0.25,
        PA_training: bool = False,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()
        self.simi_max_value = simi_max_value
        self.PA_training = PA_training

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        self.trans_type = trans_type

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            self.decoder = decoder
            self.joint_network = joint_network

            if self.trans_type =="warp-rnnt":
                try:
                    from warp_rnnt import rnnt_loss

                    self.criterion_transducer = rnnt_loss
                except ImportError:
                    raise ImportError(
                        "warp-rnnt is not installed. Please re-setup"
                        " espnet or use 'warp-transducer'"
                    )
            else:
                raise ValueError("Unsupported trans type {}, only support warp-rnnt".format(self.trans_type))

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.loss_rescaler_type = loss_rescaler_type
        self.rescaler_model_conf = rescaler_model_conf
        if loss_rescaler_type is not None:
            if "vqvae" in loss_rescaler_type:
                from espnet.nets.pytorch_backend.vqvae.conv_san_model import ConvSanEncoder
                from espnet.nets.pytorch_backend.vqvae.vq_embedding_ema import VQEmbeddingEMA
                self.vae_encoder = ConvSanEncoder(
                    idim=rescaler_model_conf["VAE_n_lower_fre_bands"],
                    odim=rescaler_model_conf["VAE_n_channels"],
                    btn_dim=rescaler_model_conf["bottleneck_dim"],
                    n_layer=rescaler_model_conf["VAE_n_layers"],
                )
                self.return_code_distances = True if loss_rescaler_type.startswith("vqvae_code_dis") else False
                self.vq_embedding = VQEmbeddingEMA(
                    n_embeddings=rescaler_model_conf["VQEmbed_n_embedding"],
                    embedding_dim=rescaler_model_conf["bottleneck_dim"],
                    init_path=rescaler_model_conf["init_codebook_path"],
                    commitment_cost=rescaler_model_conf["VQEmbed_commitment_cost"],
                    decay=rescaler_model_conf["VQEmbed_decay"],
                    epsilon=rescaler_model_conf["VQEmbed_epsilon"],
                    print_vq_prob=rescaler_model_conf["VQEmbed_print_vq_prob"],
                    return_code_distances=self.return_code_distances,
                )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        aux_speech: torch.Tensor,
        aux_speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch, )
            aux_speech: (Batch, Length2, ...)
            aux_speech_lengths: (Batch, )
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 0. rescale
        sample_weights, pat_loss_weights = self._calc_loss_rescales(speech, speech_lengths, aux_speech, aux_speech_lengths)
        sample_weights = sample_weights.unsqueeze(-1).unsqueeze(-1)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, sample_weights)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        stats["loss_weights_min"] = torch.min(sample_weights)
        stats["loss_weights_max"] = torch.max(sample_weights)
        stats["loss_weights_mean"] = torch.mean(sample_weights)
        stats["weights_list"] = sample_weights

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            if not self.PA_training:
                pat_loss_weights = None
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                raw_loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
                pat_loss_weights
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["raw_loss_transducer"] = raw_loss_transducer.detach() if raw_loss_transducer is not None else None
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        # TODO(wjm): needed to be checked
        # TODO(wjm): same problem: https://github.com/espnet/espnet/issues/4136
        # FIXME(wjm): for logger error when accum_grad > 1
        # stats["loss"] = loss.detach()
        stats["loss"] =torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_loss_rescales(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor,
            aux_speech: torch.Tensor, aux_speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            aux_speech: (Batch, Length, ...)
            aux_speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            aux_feats, aux_feats_lengths = self._extract_feats(aux_speech, aux_speech_lengths)

        def calc_vqvae_avg_probs(xs, xs_lens):
            vae_inputs = xs[:, :, :self.rescaler_model_conf["VAE_n_lower_fre_bands"]]
            h_vae_enc, h_vae_enc_mask = self.vae_encoder(vae_inputs, xs_lens)
            _, h_vq, indices = self.vq_embedding.encode(h_vae_enc)
            bb, tt, dd = h_vq.size()
            indices = torch.reshape(indices, (bb, tt))
            encodings = F.one_hot(indices, self.vq_embedding.embedding.size(0)).float()
            mask = torch.transpose(h_vae_enc_mask, 1, 2)  # -> B x T x 1
            rep = torch.sum(encodings * mask, dim=1) / torch.sum(mask, dim=1)
            return rep

        def calc_vqvae_code_dis(xs, xs_lens):
            vae_inputs = xs[:, :, :self.rescaler_model_conf["VAE_n_lower_fre_bands"]]
            h_vae_enc, h_vae_enc_mask = self.vae_encoder(vae_inputs, xs_lens)
            _, h_vq, indices, _code_dis = self.vq_embedding.encode(h_vae_enc)
            bb, tt, dd = h_vq.size()
            _code_dis = torch.reshape(_code_dis, (bb, tt, self.vq_embedding.n_embeddings))  # -> B x T x N_code
            code_mask = torch.transpose(h_vae_enc_mask, 1, 2)  # -> B x T x 1
            return _code_dis, code_mask

        raw_loss_weights = None
        if self.loss_rescaler_type.startswith("vqvae_code_dis"):
            code_dis, code_mask = calc_vqvae_code_dis(feats, feats_lengths)
            frame_code_dis = torch.min(code_dis, dim=-1, keepdim=False)[0]  # -> B x T
            code_mask = torch.squeeze(code_mask, 2)  # -> B x T
            raw_loss_weights = (frame_code_dis * code_mask).sum(1) / code_mask.sum(1)
        elif "vqvae" in self.loss_rescaler_type:
            reps = calc_vqvae_avg_probs(feats, feats_lengths)
            aux_reps = calc_vqvae_avg_probs(aux_feats, aux_feats_lengths)
            # scatter representations to calculate Cartesian product
            reps = torch.unsqueeze(reps, 1)  # B, 1, D
            aux_reps = torch.unsqueeze(aux_reps, 0)  # 1, B, D

            if self.loss_rescaler_type == "vqvae_l2norm":
                distance = torch.sum(torch.pow(reps - aux_reps, 2), dim=-1)  # B, B
                distance = torch.min(distance, dim=1)[0]  # B \in [0, 2]
                raw_loss_weights = 1.0 - (distance * 0.5)
            elif self.loss_rescaler_type == "vqvae_cosine":
                similarity = F.cosine_similarity(reps, aux_reps, dim=2, eps=1e-8)
                raw_loss_weights = torch.max(similarity, dim=1)[0]
            elif self.loss_rescaler_type == "vqvae_cosine_mean":
                similarity = F.cosine_similarity(reps, aux_reps, dim=2, eps=1e-8)
                raw_loss_weights = torch.mean(similarity, dim=1)
            elif self.loss_rescaler_type == "vqvae_cosine_normalize_mean":
                aux_reps_T = torch.transpose(aux_reps, 0, 1)  # B, 1, D
                self_simi = F.cosine_similarity(aux_reps_T, aux_reps, dim=2, eps=1e-8)
                self_simi = self_simi - torch.eye(self_simi.size(0)).to(self_simi)  # B, B
                aux_weights = torch.mean(self_simi + 1, dim=1)  # B
                aux_weights = aux_weights / torch.sum(aux_weights)  # B
                aux_weights = torch.unsqueeze(aux_weights, 0)  # 1, B
                similarity = F.cosine_similarity(reps, aux_reps, dim=2, eps=1e-8)  # B, B
                raw_loss_weights = torch.sum(similarity * aux_weights, dim=1)
            elif self.loss_rescaler_type == "vqvae_cosine_vote":
                similarity = F.cosine_similarity(reps, aux_reps, dim=2, eps=1e-8)  # N, M
                hard_vote = F.one_hot(torch.argmax(similarity, dim=0), similarity.size(0))  # M, N
                raw_loss_weights = torch.sum(hard_vote, dim=0)  # sum votes for each sample
                # half the loss value for zero-vote samples
                raw_loss_weights = torch.maximum(raw_loss_weights, torch.ones_like(raw_loss_weights) * 0.5)

        if "threshold" in self.rescaler_model_conf and self.rescaler_model_conf["threshold"] is not None:
            mask = (raw_loss_weights > self.rescaler_model_conf["threshold"]).float()
            raw_loss_weights = mask * raw_loss_weights + (1.0 - mask) * 0.1 * raw_loss_weights

        # rescale the summation of loss weights to batch size
        batch_size = feats.size(0)
        pat_loss_weights = raw_loss_weights * (batch_size / torch.sum(raw_loss_weights))
        similarity = F.sigmoid(
            (raw_loss_weights - self.simi_max_value / 2) * (10.0 / self.simi_max_value)
        )
        return similarity.detach(), pat_loss_weights.detach()

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, sample_weights=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, sample_weights, ctc=self.ctc,
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, sample_weights)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor = None,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)
            loss_weights: rescales for loos. (B, )

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )
        raw_loss_transducer = None
        if self.trans_type == "warp-transducer":
            loss_transducer = self.criterion_transducer(
                joint_out,
                target,
                t_len,
                u_len,
            )
        else:
            log_probs = torch.log_softmax(joint_out, dim=-1)

            loss_transducer = self.criterion_transducer(
                log_probs,
                target,
                t_len,
                u_len,
                reduction="none",
                blank=self.blank_id,
                gather=True,
            )
            raw_loss_transducer = loss_transducer.mean()
            if self.training:
                if self.loss_rescaler_type is not None and loss_weights is not None:
                    loss_transducer = loss_transducer * loss_weights

            loss_transducer = loss_transducer.mean()

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, raw_loss_transducer, cer_transducer, wer_transducer
