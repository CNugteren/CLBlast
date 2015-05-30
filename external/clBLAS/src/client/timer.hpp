/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#ifndef TIMER_HXX__
#define TIMER_HXX__

#include <time.h>

class timer
{
public:
  double get();
  void pause();
  void restart();
  void reset();
  void reset_delay(double delay_time);

private:
  inline double get_walltime()
  {
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<double>(ts.tv_sec) +
      static_cast<double>(ts.tv_nsec) * 1.0e-9;
  }

private:
  double init_time_;
  double elapsed_time_;

public:
    timer();
  ~timer();
}; // class timer

#endif // ifndef TIMER_HXX__
