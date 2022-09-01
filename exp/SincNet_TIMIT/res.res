SincNet(
  (conv): ModuleList(
    (0): SincConv_fast()
    (1): SparseWeights1d(
      sparsity=0.2
      (module): Conv1d(80, 60, kernel_size=(5,), stride=(1,))
    )
    (2): SparseWeights1d(
      sparsity=0.2
      (module): Conv1d(60, 60, kernel_size=(5,), stride=(1,))
    )
  )
  (bn): ModuleList(
    (0): BatchNorm1d(80, eps=983, momentum=0.05, affine=True, track_running_stats=True)
    (1): BatchNorm1d(60, eps=326, momentum=0.05, affine=True, track_running_stats=True)
    (2): BatchNorm1d(60, eps=107, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
    (1): LayerNorm()
    (2): LayerNorm()
  )
  (act): ModuleList(
    (0): LeakyReLU(negative_slope=0.2)
    (1): LeakyReLU(negative_slope=0.2)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
    (1): Dropout(p=0.0, inplace=False)
    (2): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList(
    (0): KWinners1d(channels=80, local=True, break_ties=False, n=0, percent_on=0.5, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
    (1): KWinners1d(channels=60, local=True, break_ties=False, n=0, percent_on=0.5, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
    (2): KWinners1d(channels=60, local=True, break_ties=False, n=0, percent_on=0.6, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
  )
  (ln0): LayerNorm()
)
====================

MLP_NupicTorch(
  (wx): ModuleList(
    (0): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=6420, out_features=2048, bias=True)
    )
    (1): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=2048, out_features=2048, bias=True)
    )
    (2): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=2048, out_features=2048, bias=True)
    )
  )
  (bn): ModuleList(
    (0): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    (2): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
    (1): LayerNorm()
    (2): LayerNorm()
  )
  (act): ModuleList(
    (0): LeakyReLU(negative_slope=0.2)
    (1): LeakyReLU(negative_slope=0.2)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
    (1): Dropout(p=0.0, inplace=False)
    (2): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList(
    (0): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
    (1): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
    (2): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
  )
  (ln0): LayerNorm()
)
====================

MLP(
  (wx): ModuleList(
    (0): Linear(in_features=2048, out_features=462, bias=True)
  )
  (bn): ModuleList(
    (0): BatchNorm1d(462, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
  )
  (act): ModuleList(
    (0): LogSoftmax(dim=1)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList()
)
====================

SincNet(
  (conv): ModuleList(
    (0): SincConv_fast()
    (1): SparseWeights1d(
      sparsity=0.2
      (module): Conv1d(80, 60, kernel_size=(5,), stride=(1,))
    )
    (2): SparseWeights1d(
      sparsity=0.2
      (module): Conv1d(60, 60, kernel_size=(5,), stride=(1,))
    )
  )
  (bn): ModuleList(
    (0): BatchNorm1d(80, eps=983, momentum=0.05, affine=True, track_running_stats=True)
    (1): BatchNorm1d(60, eps=326, momentum=0.05, affine=True, track_running_stats=True)
    (2): BatchNorm1d(60, eps=107, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
    (1): LayerNorm()
    (2): LayerNorm()
  )
  (act): ModuleList(
    (0): LeakyReLU(negative_slope=0.2)
    (1): LeakyReLU(negative_slope=0.2)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
    (1): Dropout(p=0.0, inplace=False)
    (2): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList(
    (0): KWinners1d(channels=80, local=True, break_ties=False, n=0, percent_on=0.5, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
    (1): KWinners1d(channels=60, local=True, break_ties=False, n=0, percent_on=0.5, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
    (2): KWinners1d(channels=60, local=True, break_ties=False, n=0, percent_on=0.6, boost_strength=1.5, boost_strength_factor=1.0, k_inference_factor=1.0, duty_cycle_period=250)
  )
  (ln0): LayerNorm()
)
====================

MLP_NupicTorch(
  (wx): ModuleList(
    (0): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=6420, out_features=2048, bias=True)
    )
    (1): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=2048, out_features=2048, bias=True)
    )
    (2): SparseWeights(
      sparsity=0.2
      (module): Linear(in_features=2048, out_features=2048, bias=True)
    )
  )
  (bn): ModuleList(
    (0): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    (2): BatchNorm1d(2048, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
    (1): LayerNorm()
    (2): LayerNorm()
  )
  (act): ModuleList(
    (0): LeakyReLU(negative_slope=0.2)
    (1): LeakyReLU(negative_slope=0.2)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
    (1): Dropout(p=0.0, inplace=False)
    (2): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList(
    (0): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
    (1): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
    (2): KWinners(n=2048, percent_on=0.7, boost_strength=1.0, boost_strength_factor=0.9, k_inference_factor=1.0, duty_cycle_period=250, break_ties=False)
  )
  (ln0): LayerNorm()
)
====================

MLP(
  (wx): ModuleList(
    (0): Linear(in_features=2048, out_features=462, bias=True)
  )
  (bn): ModuleList(
    (0): BatchNorm1d(462, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
  )
  (ln): ModuleList(
    (0): LayerNorm()
  )
  (act): ModuleList(
    (0): LogSoftmax(dim=1)
  )
  (drop): ModuleList(
    (0): Dropout(p=0.0, inplace=False)
  )
  (kwinners): ModuleList()
)
====================

