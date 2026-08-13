# MO localization: текущая архитектура

**Состояние:** MO-функциональность слита в `genisom_roamerx_open` на 2026-08-08.  
**Область:** vendor-уровень локализации и интеграции с `robot_navigo`.

Этот поддерживаемый vendor-репозиторий содержит MO-версии пакетов `localization`
и `robot_navigo`. Прикладная оркестрация, safety gate, web state и управление
находятся в соседнем репозитории `security-robot-main`.

## 1. Поток данных

```text
/front_lidar
  -> pointcloud_self_filter
  -> /front_lidar_filtered
  -> localization (planar NDT_OMP)

/odom/mc_odom
  -> motor_odom_deduplicator
  -> /odom/mc_odom_unique
  -> localization predictor: linear twist + causal pose-increment reconciliation

/front_lidar/imu
  -> localization predictor: gyro yaw rate + IMU data

map.pcd -> /load_map_service -> localization
/map -------------------------> map pose gate
/initialpose -----------------> initialization/relocalization
/gnss/fix --------------------> optional recovery seed only
```

Выход:

```text
localization
  -> /localization_info_health
  -> /odom/localization_odom
  -> /odom/fused_odom
  -> /aligned_points
  -> /global_map_points
  -> /status
  -> /mo_tf

lidar_tf -> /mo_tf_static
```

## 2. Ноды

| Нода | Назначение |
|---|---|
| `/pointcloud_self_filter` | Удаляет корпус и заднюю защитную стойку из LiDAR cloud |
| `/motor_odom_deduplicator` | Пропускает первый пакет каждого строго возрастающего motor-odom timestamp |
| `/localization` | IMU initialization, predictor, NDT tracking/recovery, health и TF |
| `/lidar_tf` | Публикует статический `base_link -> livox_frame` |

## 3. Предиктор и NDT

Текущая схема не использует абсолютную motor-odom pose как глобальную позу:

- translation интегрируется из motor `twist`;
- последовательные приращения motor pose устраняют пропущенный путь только
  после пауз timestamp длиннее twist timeout; в штатном потоке predictor
  остаётся twist-driven, произвольный origin не используется;
- yaw интегрируется из IMU gyro;
- motor yaw плавно отслеживает преобразование fixed-осей vendor odom в fused frame,
  но не заменяет IMU-ориентацию;
- `enable_internal_odom_ukf: false`, второй odometry UKF отключен;
- NDT корректирует предсказанную позу относительно PCD map;
- deskew выключен;
- локализация планарная: `x/y/yaw`.

Текущие вычислительные параметры исходника:

| Параметр | Значение |
|---|---:|
| Main NDT threads | 4 |
| Recovery NDT threads | 2 |
| ROS MultiThreadedExecutor | 3 |
| NDT resolution | 0.5 m |
| Input voxel | 0.30 m |
| Main NDT max rate | 4 Hz |
| Fused odom publish rate | 50 Hz |

Основная NDT-коррекция проверяется по fitness, максимальному скачку `x/y/yaw`,
согласованности с fused predictor и occupancy map. Фоновый recovery сначала
ищет позу около causal fused-odom seed, затем проверяет её на 6 последовательных
NDT-сканах без изменения рабочего фильтра. GNSS, если включён отдельным site
config, используется только как запасной seed после неудачи odom-поиска. При
неподвижном роботе recovery ограничен `0.20 m` и `5 deg`.

## 4. Статусы

Канонический topic: `/localization_info_health`, тип `robots_dog_msgs/msg/Localization`.

| Status | Значение | Autonomy |
|---:|---|---|
| 0 | Initializing | запрещена |
| 1 | Relocalizing | запрещена |
| 2 | Relocalization succeeded, переход | запрещена |
| 3 | Normal tracking | разрешена внешним safety gate |
| 4 | Lost/degraded/extrapolating | запрещена |

Status `4` может содержать удержанную или предсказанную позу для диагностики. Это не делает ее надежной и не разрешает движение.

`publish_vendor_localization_info: false`, поэтому `/localization_info` не является рабочим health-интерфейсом MO.

## 5. TF

MO использует изолированные topics:

```text
/mo_tf:        map -> odom -> base_link
/mo_tf_static: base_link -> livox_frame
```

Launch-файлы `robot_navigo` remap-ят `/tf` в `/mo_tf`, а `/tf_static` в `/mo_tf_static`. Legacy `/robot_tf` может продолжать публиковать одноименные frames в `/tf*`, но Nav2 MO не должен их потреблять.

## 6. ROS-интерфейсы

### Входы

| Имя | Тип |
|---|---|
| `/front_lidar` | `sensor_msgs/msg/PointCloud2` |
| `/front_lidar_filtered` | `sensor_msgs/msg/PointCloud2` |
| `/front_lidar/imu` | `sensor_msgs/msg/Imu` |
| `/odom/mc_odom` | `nav_msgs/msg/Odometry` |
| `/odom/mc_odom_unique` | `nav_msgs/msg/Odometry` |
| `/map` | `nav_msgs/msg/OccupancyGrid` |
| `/initialpose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| `/gnss/fix` | `sensor_msgs/msg/NavSatFix` (только при `use_gnss_recovery: true`) |
| `/load_map_service` | `robots_dog_msgs/srv/LoadMap` |

### Выходы

| Имя | Тип | Назначение |
|---|---|---|
| `/localization_info_health` | `robots_dog_msgs/msg/Localization` | Канонический status и map pose |
| `/odom/localization_odom` | `nav_msgs/msg/Odometry` | LiDAR/map pose |
| `/odom/fused_odom` | `nav_msgs/msg/Odometry` | Непрерывный predictor output |
| `/aligned_points` | `sensor_msgs/msg/PointCloud2` | Диагностика регистрации |
| `/global_map_points` | `sensor_msgs/msg/PointCloud2` | PCD map cloud |
| `/status` | `localization/msg/ScanMatchingStatus` | Детальная диагностика NDT |
| `/mo_tf`, `/mo_tf_static` | `tf2_msgs/msg/TFMessage` | Канонический TF для Nav2 MO |

Nav2 публикует команды в `/cmd_vel_autonomy`; дальнейший safety gate реализован в `security-robot-mo`, а не в этом vendor-репозитории.

## 7. Сборка на роботе

Отдельный MO overlay после слияния не используется:

```bash
source /opt/ros/humble/setup.bash
cd /home/jszr/genisom_roamerx_open
source install/setup.bash 2>/dev/null || true
colcon build --symlink-install --packages-up-to localization robot_navigo
```

Runtime-проверка выполняется из прикладного репозитория:

```bash
cd /home/jszr/security-robot-main
source scripts/robot/env.sh
ros2 pkg prefix localization
ros2 pkg prefix robot_navigo
ros2 pkg executables localization
```

`localization` должен разрешаться в `genisom_roamerx_open/install`, а executables
должны включать `localization_node`, `motor_odom_deduplicator_node` и
`pointcloud_self_filter_node`. Prefix из старого `install_mo_overlay` означает,
что shell всё ещё содержит устаревшее окружение.

## 8. Известные риски

1. `/odom/mc_odom` имеет два известных publisher (`ecal2ros2`, `dog_task`). Deduplicator проверяет timestamp, но не фиксирует конкретный publisher. В полевых bag встречались 2–3-секундные серии повторов одного timestamp; predictor удерживает скорость лишь 0.35 s и затем причинно догоняет измеренное приращение после прихода свежего timestamp.
2. LiDAR timestamps запаздывают относительно ROS now; costmap может отбрасывать облака как более старые, чем TF cache.
3. В слабой геометрии и на краю карты NDT может потерять однозначный минимум.
4. Main/recovery NDT, Nav2 и Zenoh конкурируют за CPU; Foxglove усиливает нагрузку подписками на облака и TF.
5. Topic `/goal_pose` не подтверждает доставку. Для production-команд предпочтителен action `/navigate_to_pose`.

Полный прикладной контракт для web stack находится в:

```text
security-robot-main/docs/LOCALIZATION_WEB_STACK_INTERFACE_RU.md
```
