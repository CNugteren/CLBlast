#!/bin/bash

FUNCTIONS=(gemm trmm trsm syrk syr2k gemv symv)
ALL_PRECISIONS=(s c)
ALL_OPTIONS=(order transA transB side uplo diag M N K incx incy offA offBX offCY)

# list of supported options for each function: gemm, trmm, trsm, syrk, ssyr2k, gemv, symv
FUNC_OPTIONS=( "order transA transB M N K"
               "order transA side uplo diag M N"
               "order transA side uplo diag M N"
               "order transA uplo N K"
               "order  transA uplo N K"
               "order transA M N"
               "order uplo N" )

# all options space: precision, order, transA, transB, side, uplo, unit, M, N, K
ALL_OPTION_VALUES=( "row column"
                    "n t c"
                    "n t c"
                    "left right"
                    "upper lower"
                    "unit nonunit"
                    "15 16 64"
                    "15 16 64"
                    "15 16 64"
                    "1"
                    "1"
                    "128"
                    "256"
                    "512" )

REPORT_FILE="ktest_report.dat"

PREV_KERNEL=""
REMAINING_OPTSTR=
CMDLINE=
FUNCTION_INDEX=

forward_options_and_call_test()
{
    local optidx=$1
    local precision=$2
    local optstr=${REMAINING_OPTSTR[@]}
    local ret=0
    local stat=0
    local cmdline=
    local msg=
    local err_msg=
    
    for opt in ${optstr[@]}
    do
        REMAINING_OPTSTR=${REMAINING_OPTSTR[@]##$opt}
        echo ${FUNC_OPTIONS[$FUNCTION_INDEX]} | grep $opt > /dev/null
        if [ $? -eq 0 ]
        then
            break
        fi

        let "optidx += 1"
    done

    # make test and call if no more options to forward, or go further in the option list
    if [ $optidx == ${#ALL_OPTIONS[@]} ]
    then 
        cmdline="--function "$PRECISION${FUNCTIONS[$FUNCTION_INDEX]}" ${CMDLINE[@]}"
        echo ${cmdline[@]}
        ./make-ktest ${cmdline[@]}
        stat=$?
        err_msg="[ERROR]: make-ktest has failed!"
        if [ $stat -eq 0 ]
        then
            # check if the kernel is not the same as the last one
            kernel=`cat *.cl`
            if [ "${kernel[*]}" == "${LAST_KERNEL[*]}" ]
            then
                echo "Critical error, just the same kernel has been already generated!"
                return 1
            fi
        fi

        if [ $stat -eq 0 ]
        then
            g++ -o test ktest.cpp -I$AMDAPPSDKROOT/include -lOpenCL
            stat=$?
            err_msg="[ERROR]: test compilation has failed!"=
        fi

        if [ $stat -eq 0 ]
        then
            msg=`./test 2>&1`
            stat=$?
        fi

        if [ $stat -eq 0 ]
        then
            time_msg=${msg/Correctness*/""}
            msg=${msg##$time_msg}
            echo $time_msg
            echo $msg
            echo $msg | grep "passed" > /dev/null
            stat=$?
        fi

        if [ $stat -ne 0 ]
        then
            echo $err_msg
            echo ${cmdline[@]} >> $REPORT_FILE.tmp
        fi
    else
        local OPTION=${ALL_OPTIONS[$optidx]}
        local OPTION_VALUES=${ALL_OPTION_VALUES[$optidx]}

        let "optidx += 1"
        cmdline=${CMDLINE[@]}
        
        for val in ${OPTION_VALUES[@]}
        do
            CMDLINE=${cmdline[@]}" --$OPTION ""$val"
            (forward_options_and_call_test $optidx)
            ret=$?
            if [ $ret -ne 0 ]
            then
                break
            fi

            LAST_KERNEL=$kernel
            rm -f *.cl > /dev/null
        done
    fi

    return $ret
}

rm -f *.cl > /dev/null

> $REPORT_FILE.tmp

# test the main funtional
for ((i = 0; i < ${#FUNCTIONS[@]}; i++))
do
    FUNCTION_INDEX=$i
    for PRECISION in ${ALL_PRECISIONS[@]}
    do
        if [[ ${FUNCTIONS[$i]} == symv && $PRECISION == c ]]
        then
            continue
        fi

        CMDLINE=""
        REMAINING_OPTSTR=${ALL_OPTIONS[@]}
        forward_options_and_call_test 0
    done
done

echo ==========================================================================================

# test increment and offset arguments

FUNC_OPTIONS=( "order transA transB M N K offA offCY"
               "order transA side uplo diag M N offA offBX"
               "order transA side uplo diag M N offA"
               "order transA uplo N K offA offCY"
               "order  transA uplo N K offA offBX offCY"
               "order transA M N offA incx incy offA"
               "order uplo N offA incx incy offA" )

ALL_OPTION_VALUES=( "row column"
                    "n"
                    "n"
                    "left"
                    "upper"
                    "nonunit"
                    "64"
                    "64"
                    "64"
                    "1 3 7"
                    "1 5 9"
                    "128"
                    "256"
                    "512" )

for ((i = 0; i < ${#FUNCTIONS[@]}; i++))
do
    FUNCTION_INDEX=$i
    for PRECISION in ${ALL_PRECISIONS[@]}
    do
        if [[ ${FUNCTIONS[$i]} == symv && $PRECISION == c ]]
        then
            continue
        fi

        CMDLINE=""
        REMAINING_OPTSTR=${ALL_OPTIONS[@]}
        forward_options_and_call_test 0
    done
done

# complete the report
report=`cat $REPORT_FILE.tmp`
nr_fails=`cat $REPORT_FILE.tmp | wc -l`
if [ $nr_fails == 0 ]
then
    echo "All tests passed" > $REPORT_FILE
else
    echo "Failed cases:" > $REPORT_FILE
    echo "-----------------------------------------------------------------" >> $REPORT_FILE
    cat $REPORT_FILE.tmp >> $REPORT_FILE
    echo "-----------------------------------------------------------------" >> $REPORT_FILE
    echo "Total number of failed cases: $nr_fails" >> $REPORT_FILE
fi

rm $REPORT_FILE.tmp
