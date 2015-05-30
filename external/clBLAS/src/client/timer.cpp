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

#include "ctimer.h"
#include "timer.hpp"

timer::
timer() : elapsed_time_(0.0)
{
  init_time_ = get_walltime();
}

timer::
~timer()
{
}

double
timer::
get()
{
  return elapsed_time_ + get_walltime() - init_time_;
}

void
timer::
pause()
{
  elapsed_time_ = get();
}

void
timer::
restart()
{
  init_time_ = get_walltime();
}

void
timer::
reset()
{
  elapsed_time_ = 0.0;
  init_time_ = get_walltime();
}

void
timer::
reset_delay(double delay_time)
{
  reset();
  elapsed_time_ = delay_time;
}

Timer CreateTimer()
{
  Timer local_timer = new timer();
  return local_timer;
}

void DeleteTimer(Timer timer)
{
  delete timer;
}

double GetTime(Timer timer)
{
  return timer->get();
}

void ResetTimer(Timer timer)
{
  timer->reset();
}

void RestartTimer(Timer timer)
{
  timer->restart();
}

void PauseTimer(Timer timer)
{
  timer->pause();
}

void ResetDelayTimer(Timer timer, double delay_time)
{
  timer->reset_delay(delay_time);
}

