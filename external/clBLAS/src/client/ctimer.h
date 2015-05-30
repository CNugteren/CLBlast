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

#ifndef C_TIMER_HXX__
#define C_TIMER_HXX__

#if defined(__cplusplus)
typedef class timer *Timer;
#else
typedef struct timer *Timer;
#endif

#if defined(__cplusplus)
extern "C" {
#endif

extern Timer CreateTimer();
extern void DeleteTimer(Timer timer);
extern double GetTime(Timer timer);
extern void PauseTimer(Timer timer);
extern void RestartTimer(Timer timer);
extern void ResetTimer(Timer timer);
extern void ResetDelayTimer(Timer timer, double delay_time);

#if defined(__cplusplus)
}
#endif

#endif // ifndef C_TIMER_HXX__
