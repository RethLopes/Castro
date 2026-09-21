# This script runs a compiled detonation multiple times
# Each iteration is run with a random sampling of StarLib rates

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 \"<executable command>\" <inputs file> <num_runs> <output_folder_name>"
    exit 1
fi

EXEC="$1"
INPUTS="$2"
NRUNS="$3"
OUTDIR="$4"
 
if [ ! -f "$INPUTS" ]; then
    echo "Error: inputs file '$INPUTS' not found"
    exit 1
fi

# create output directory and log file
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/summary.txt"
touch $LOG

# This function checks whether a run was successful
check_run() {
    local err_file="$1"
 
    if [ ! -s "$err_file" ]; then
        # empty error file -- clean success
        rm -f "$err_file"
        echo "SUCCESS" | tee -a "$LOG"
    elif grep -q "detonation has reached the edge of the domain" "$err_file"; then
        # reached edge of domain" -- also a success
        rm -f "$err_file"
        echo "SUCCESS (edge of domain)" | tee -a "$LOG"
    else
        # a real error -- keep the err file for inspection
        echo "ERROR" | tee -a "$LOG"
    fi
}

#First run the median case
MED_DIR="$OUTDIR/run_median"
mkdir -p "$MED_DIR"

printf "Median Run, STATUS: " | tee -a "$LOG"
MED_LOG="${OUTDIR}/det_log_med.txt"
MED_ERR="${OUTDIR}/det_err_med.txt"
eval "$EXEC $INPUTS network.starlib_seed=-1" > "${MED_LOG}" 2> "${MED_ERR}"
check_run "$MED_ERR"

mv det_x_plt* "${MED_DIR}"
rm -rf det_x_chk*

# create an map to keep track of used seeds
declare -A used_seeds

for (( i=1; i<=NRUNS; i++)); do

    RUN_DIR="${OUTDIR}/run_${i}"
    mkdir -p "$RUN_DIR"

    while true;  do
        SEED=$RANDOM
        if [ -z "${used_seeds[$SEED]}" ]; then
            used_seeds[$SEED]=1
            break
        fi
    done

    printf "Run %d: %s, STATUS:" "$i" "$SEED" | tee -a "$LOG"
    RUN_LOG="${OUTDIR}/det_log_${i}.txt"
    RUN_ERR="${OUTDIR}/det_err_${i}.txt"
    eval "$EXEC $INPUTS network.starlib_seed=${SEED}" > "${RUN_LOG}" 2> "${RUN_ERR}"
    check_run "$RUN_ERR"

    mv det_x_plt* "${RUN_DIR}"
    rm -rf det_x_chk*

done