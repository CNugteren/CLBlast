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


#include <string.h>         /* strcmp */
#include <stdlib.h>         /* atoi, strtol */
#include <stdio.h>          /* printf */

#include <cmdline.h>

static const char *testUsage =
    "<M N K> [--seed s] [--alpha a] [--beta b] "
    "[--alpha-real a] [--beta-real b] [--alpha-imag a] [--beta-imag b] "
    "[--use-images f] [--device dev] [--queues n]\n"
    "\n"
    "seed - seed for the random number generator"
    "\n"
    "alpha - alpha multiplier"
    "\n"
    "beta - beta multiplier"
    "\n"
    "alpha-real - alpha multiplier real part"
    "\n"
    "beta-real - beta multiplier real part"
    "\n"
    "alpha-imag - alpha-multiplier imaginary part"
    "\n"
    "beta-imag - beta-multiplier imaginary part"
    "\n"
    "use-images - allow the library to use images for computing"
    "\n"
    "device - device to run the test on, 'cpu' or 'gpu'(default)"
    "\n"
    "queues - number of command queues to use"
    "\n"
    "Parameters defined through the command line are kept over the whole "
    "set of custom test cases. The use-images parameter value is ignored if "
    "the target device is CPU\n\n";

typedef struct SetterArg {
    TestParams *params;
    const char *arg;
    long extra;
} SetterArg;

typedef struct CmdLineOpt {
    const char *name;
    unsigned int flagToSet;
    int (*setter)(SetterArg*);
    long setterExtra;
} CmdLineOpt;

enum {
    MULT_ALPHA = 0x01,
    MULT_BETA = 0x02,
    MULT_REAL_ONLY = 0x04,
    MULT_IMAG_ONLY = 0x08
};

static int
doParseCmdLine(
    int argc,
    char *argv[],
    const CmdLineOpt *opts,
    unsigned int nrOpts,
    TestParams *params)
{
    int i = 1, j = 0;
    int ret = 0;
    const CmdLineOpt *currOpt;
    const char *currArg;
    SetterArg sarg = {params, NULL, 0};

    do {
        currArg = (const char*)argv[i];
        i++;

        if (currArg[0] != '-') {
            // some of size arguments
            switch (j) {
            case 0:
                params->M = atoi(currArg);
                params->optFlags |= SET_M;
                break;
            case 1:
                params->N = atoi(currArg);
                params->optFlags |= SET_N;
                break;
            case 2:
                params->K = atoi(currArg);
                params->optFlags |= SET_K;
                break;
            }
            j++;
            continue;
        }
        else if (currArg[1] != '-') {
            // it can be some parameter of a used test framework, skip it
            j = 0;
            continue;
        }

        j = 0;

        for (currOpt = opts; currOpt < opts + nrOpts; currOpt++) {
            if (!strcmp(currOpt->name, &currArg[2])) {
                if (i == argc) {
                    printf("Error: parameter '%s' is not specified!\n",
                           currOpt->name);
                    ret = -1;
                }
                else {
                    sarg.arg = argv[i++];
                    sarg.extra = currOpt->setterExtra;
                    ret = currOpt->setter(&sarg);
                    params->optFlags |= currOpt->flagToSet;
                }
                break;
            }
        }
    } while ((i < argc) && !ret);

    return ret;
}

static int
setSeed(SetterArg *sarg)
{
    sarg->params->seed = atoi(sarg->arg);

    return 0;
}

static int
setMult(SetterArg *sarg)
{
    ComplexLong *mult;
    long val;
    char *end;
    long flags = sarg->extra;

    mult = (flags & MULT_BETA) ? &sarg->params->beta :
                                 &sarg->params->alpha;
    mult->re = 0;
    mult->imag = 0;

    val = strtol(sarg->arg, &end, 10);
    if (!(flags & MULT_IMAG_ONLY)) {
        mult->re = val;
    }
    if (!(flags & MULT_REAL_ONLY)) {
        mult->imag = val;
    }

    return 0;
}

static int
setDevice(SetterArg *sarg)
{
    if (!strcmp(sarg->arg, "cpu")) {
        sarg->params->devType = CL_DEVICE_TYPE_CPU;
        sarg->params->devName = NULL;
        return 0;
    }
    if (!strcmp(sarg->arg, "gpu")) {
        sarg->params->devType = CL_DEVICE_TYPE_GPU;
        sarg->params->devName = NULL;
        return 0;
    }
    sarg->params->devName = sarg->arg;

    return 0;
}

static int
setNumCommandQueues(SetterArg *sarg)
{
    sarg->params->numCommandQueues = atoi(sarg->arg);

    return 0;
}

static const CmdLineOpt opts[] = {
    {"seed", SET_SEED, setSeed, 0},
    {"alpha", SET_ALPHA, setMult, MULT_ALPHA | MULT_REAL_ONLY},
    {"beta", SET_BETA, setMult, MULT_BETA | MULT_REAL_ONLY},
    {"alpha-real", SET_ALPHA, setMult, MULT_ALPHA | MULT_REAL_ONLY},
    {"alpha-imag", SET_ALPHA, setMult, MULT_ALPHA | MULT_IMAG_ONLY},
    {"beta-real", SET_BETA, setMult, MULT_BETA | MULT_REAL_ONLY},
    {"beta-imag", SET_BETA, setMult, MULT_BETA | MULT_IMAG_ONLY},
    {"device", SET_DEVICE_TYPE, setDevice, 0},
    {"queues", SET_NUM_COMMAND_QUEUES, setNumCommandQueues, 0},
};
static const unsigned int nrOpts = sizeof(opts) / sizeof(CmdLineOpt);

int
parseBlasCmdLineArgs(
    int argc,
    char *argv[],
    TestParams *params)
{
    return doParseCmdLine(argc, argv, opts, nrOpts, params);
}

void
printUsage(const char *appName)
{
    printf("%s %s\n", appName, testUsage);
}

void
parseEnv(TestParams *params)
{
    const char *str;
    int createImages = 0;

    str = getenv("AMD_CLBLAS_GEMM_IMPLEMENTATION");
    if ((str != NULL) && (strcmp(str, "1") == 0)) {
        createImages = 1;
    }
    str = getenv("AMD_CLBLAS_TRMM_IMPLEMENTATION");
    if ((str != NULL) && (strcmp(str, "1") == 0)) {
        createImages = 1;
    }
    str = getenv("AMD_CLBLAS_TRSM_IMPLEMENTATION");
    if ((str != NULL) && (strcmp(str, "1") == 0)) {
        createImages = 1;
    }

	params->optFlags	= NO_FLAGS;
    if (createImages) {
        params->optFlags |= SET_USE_IMAGES;
    }
}
