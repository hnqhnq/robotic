# E6 失败模式（自动生成）

## e3-base-h8
- episodes: 500, fail: 82
- fail tags: {'timeout': 38, 'timeout_stall': 44}
- mean stall (fail): 0.352

## e3-tar-h8
- episodes: 500, fail: 58
- fail tags: {'timeout_stall': 40, 'timeout': 18}
- mean stall (fail): 0.467

## e4-base-h4
- episodes: 500, fail: 125
- fail tags: {'timeout': 54, 'timeout_stall': 71}
- mean stall (fail): 0.419

## e4-tar-h4
- episodes: 500, fail: 99
- fail tags: {'timeout_stall': 84, 'timeout': 15}
- mean stall (fail): 0.572

## e4-base-h2
- episodes: 500, fail: 186
- fail tags: {'timeout': 49, 'timeout_stall': 137}
- mean stall (fail): 0.499

## e4-tar-h2
- episodes: 500, fail: 155
- fail tags: {'timeout_stall': 137, 'timeout': 18}
- mean stall (fail): 0.613

## 配对 e3-base-h8 vs e3-tar-h8
- paired n: 500
- outcomes: {'both_ok': 377, 'base_only': 41, 'tar_only': 65, 'both_fail': 17, 'fail_timeout|timeout_stall': 3, 'fail_timeout_stall|timeout_stall': 11, 'fail_timeout|timeout': 2, 'fail_timeout_stall|timeout': 1}
- baseline fail tags: {'timeout': 38, 'timeout_stall': 44}
- TAR fail tags: {'timeout_stall': 40, 'timeout': 18}

## 配对 e4-base-h4 vs e4-tar-h4
- paired n: 500
- outcomes: {'both_ok': 310, 'base_only': 65, 'tar_only': 91, 'both_fail': 34, 'fail_timeout_stall|timeout': 4, 'fail_timeout_stall|timeout_stall': 17, 'fail_timeout|timeout_stall': 10, 'fail_timeout|timeout': 3}
- baseline fail tags: {'timeout': 54, 'timeout_stall': 71}
- TAR fail tags: {'timeout_stall': 84, 'timeout': 15}

## 配对 e4-base-h2 vs e4-tar-h2
- paired n: 500
- outcomes: {'tar_only': 97, 'both_ok': 248, 'base_only': 66, 'both_fail': 89, 'fail_timeout_stall|timeout': 6, 'fail_timeout|timeout': 3, 'fail_timeout_stall|timeout_stall': 58, 'fail_timeout|timeout_stall': 22}
- baseline fail tags: {'timeout': 49, 'timeout_stall': 137}
- TAR fail tags: {'timeout_stall': 137, 'timeout': 18}
