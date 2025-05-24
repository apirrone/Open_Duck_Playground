- removing `accelerometer[0] += 1.3` in inference seem to help ?
  - [] Maybe should remove in joystick too. Check on real robot
- ticking `imitation_i` only when not zero command help to get a nice standing position, and does not degrade catching on push. Nice
  - [] Try doing that also during training, and remove stand still cost ?
- Still trouble detecting slow lean. Only reacts on pretty strong pushes


TRY :
- no stand still cost + head mix + less randomization
- more realistic push
- less noise on imu (gyro and accelero)
