# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import torch


@torch.jit.script
def boost_activations(x, duty_cycles, boost_strength: float):
    """
    Boosting as documented in :meth:`kwinners` would compute
      x * torch.exp((target_density - duty_cycles) * boost_strength)
    but instead we compute
      x * torch.exp(-boost_strength * duty_cycles)
    which is equal to the former value times a positive constant, so it will
    have the same ranked order.

    :param x:
      Current activity of each unit.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :return:
         A tensor representing the boosted activity
    """
    if boost_strength > 0.0:
        return x.detach() * torch.exp(-boost_strength * duty_cycles)
    else:
        return x.detach()


@torch.jit.script
def kwinners(x, duty_cycles, k: int, boost_strength: float, break_ties: bool = False,
             relu: bool = False, inplace: bool = False):
    """
    A simple K-winner take all function for creating layers with sparse output.

    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as:

    .. math::
        boostFactors = \\exp(-boostStrength \\times (dutyCycles - targetDensity))

    Intuitively this means that units that have been active (i.e. in the top-k)
    at the target activation level have a boost factor of 1, meaning their
    activity is not boosted. Columns whose duty cycle drops too much below that
    of their neighbors are boosted depending on how infrequently they have been
    active. Unit that has been active more than the target activation level
    have a boost factor below 1, meaning their activity is suppressed and
    they are less likely to be in the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the duty_cycles via an exponential function::

            boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> duty_cycles
                   |
              target_density

    :param x:
      Current activity of each unit, optionally batched along the 0th dimension.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners

    :param inplace:
      Whether to modify x in place

    :return:
      A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        indices = boosted.topk(k=k, dim=1, sorted=False)[1]
        off_mask = torch.ones_like(boosted, dtype=torch.bool)
        off_mask.scatter_(1, indices, 0)

        if relu:
            off_mask.logical_or_(boosted <= 0)
    else:
        threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                     keepdim=True)[0]

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)


@torch.jit.script
def kwinners2d(x, duty_cycles, k: int, boost_strength: float, local: bool = True,
               break_ties: bool = False, relu: bool = False,
               inplace: bool = False):
    """
    A K-winner take all function for creating Conv2d layers with sparse output.

    If local=True, k-winners are chosen independently for each location. For
    Conv2d inputs (batch, channel, H, W), the top k channels are selected
    locally for each of the H X W locations. If there is a tie for the kth
    highest boosted value, there will be more than k winners.

    The boost strength is used to compute a boost factor for each unit
    represented in x. These factors are used to increase the impact of each unit
    to improve their chances of being chosen. This encourages participation of
    more columns in the learning process. See :meth:`kwinners` for more details.

    :param x:
      Current activity of each unit.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units across the channels will be allowed to
      remain, the rest are set to zero.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :param local:
      Whether or not to choose the k-winners locally (across the channels at
      each location) or globally (across the whole input and across all
      channels).

    :param break_ties:
      Whether to use a strict k-winners. Using break_ties=False is faster but
      may occasionally result in more than k active units.

    :param relu:
      Whether to simulate the effect of applying ReLU before KWinners.

    :param inplace:
      Whether to modify x in place

    :return:
         A tensor representing the activity of x after k-winner take all.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        if local:
            indices = boosted.topk(k=k, dim=1, sorted=False)[1]
            off_mask = torch.ones_like(boosted, dtype=torch.bool)
            off_mask.scatter_(1, indices, 0)
        else:
            shape2 = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            indices = boosted.view(shape2).topk(k, dim=1, sorted=False)[1]
            off_mask = torch.ones(shape2, dtype=torch.bool, device=x.device)
            off_mask.scatter_(1, indices, 0)
            off_mask = off_mask.view(x.shape)

        if relu:
            off_mask.logical_or_(boosted <= 0)
    else:
        if local:
            threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                         keepdim=True)[0]
        else:
            threshold = boosted.view(x.shape[0], -1).kthvalue(
                x.shape[1] * x.shape[2] * x.shape[3] - k + 1, dim=1)[0]
            threshold = threshold.view(x.shape[0], 1, 1, 1)

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)

@torch.jit.script
def kwinners1d(x, duty_cycles, k: int, boost_strength: float, local: bool = True,
               break_ties: bool = False, relu: bool = False,
               inplace: bool = False):
    """
    疎な出力を持つConv2d層を作成するためのK-winner take all 関数.

    local=True の場合, k-winner は各位置で独立に選択される.
    Conv1d 入力 (batch, channel, L) に対して、Lの各位置で上位 k チャンネルが局所的に選択される。
    k 番目に高いブースト値が同点であれば、勝者は k 人以上となる。

    ブースト強度は、x で表される各ユニットのブースト係数を計算するために使用される。
    これらの係数は、各ユニットが選ばれる可能性を向上させるために、その影響を増大させるために使用される。
    これにより、より多くの列が学習プロセスに参加することが奨励される。詳細は :meth:`kwinners` を参照すること。

    :param x: :param x: 各ユニットの現在の活動量．

    :param duty_cycles: 各ユニットの現在の活動量．param duty_cycles: 各ユニットの平均的な負荷サイクル．

    :param k: param k: チャンネル間で上位 k ユニットのアクティビティを許可し、残りは 0 に設定される。

    :param boost_strength: param boost_strength: ブースト強度が0.0の場合、xに影響を与えません。

    :param local: :param local: k-winnersをローカルに（各位置のチャンネルを横断して）選択するか、
    グローバルに（入力全体と全チャンネルを横断して）選択するかを指定します。

    :param break_ties: param break_ties: k-winnersを厳密に使用するかどうか．break_ties=False にすると高速になりますが，
    場合によってはk個以上のユニットがアクティブになることがあります．

    :param relu: param relu: KWinnersの前にReLUを適用する効果をシミュレートするかどうかを指定します．

    :param inplace: param inplace: xを定位置で修正するかどうか :return: k-winner take all後のxの活性度を表すテンソル．

    """
    if k == 0:
        return torch.zeros_like(x)

    boosted = boost_activations(x, duty_cycles, boost_strength)

    if break_ties:
        if local:
            indices = boosted.topk(k=k, dim=1, sorted=False)[1]
            off_mask = torch.ones_like(boosted, dtype=torch.bool)
            off_mask.scatter_(1, indices, 0)
        else:
            shape2 = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            indices = boosted.view(shape2).topk(k, dim=1, sorted=False)[1]
            off_mask = torch.ones(shape2, dtype=torch.bool, device=x.device)
            off_mask.scatter_(1, indices, 0)
            off_mask = off_mask.view(x.shape)

        if relu:
            off_mask.logical_or_(boosted <= 0)
    else:
        if local:
            threshold = boosted.kthvalue(x.shape[1] - k + 1, dim=1,
                                         keepdim=True)[0]
        else:
            threshold = boosted.view(x.shape[0], -1).kthvalue(
                x.shape[1] * x.shape[2] * x.shape[3] - k + 1, dim=1)[0]
            threshold = threshold.view(x.shape[0], 1, 1, 1)

        if relu:
            threshold.clamp_(min=0)
        off_mask = boosted < threshold

    if inplace:
        return x.masked_fill_(off_mask, 0)
    else:
        return x.masked_fill(off_mask, 0)

__all__ = [
    "kwinners",
    "kwinners2d",
    "kwinners1d",
]
