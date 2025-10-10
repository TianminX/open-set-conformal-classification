#!/bin/bash

# Parameters
total_images=202599
<<<<<<< Updated upstream
batch_size=500
=======
#total_images=10
batch_size=1000
>>>>>>> Stashed changes

# Calculate the number of batches
num_batches=$(( (total_images + batch_size - 1) / batch_size ))
batch_list=$(seq 1 $num_batches)
#batch_list=$(seq 1 2)

# Slurm parameters
<<<<<<< Updated upstream
MEMO=3G                             # Memory required (1 GB)
TIME=00-00:30:00                    # Time required (2 h)
=======
MEMO=1G                             # Memory required (1 GB)
TIME=00-02:00:00                    # Time required (2 h)
>>>>>>> Stashed changes
CORE=1                              # Cores required (1)

# Assemble order                                               prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs/celeb_preprocess"
mkdir -p $LOGS

comp=0
incomp=0

OUT_DIR="../../data/celebrity/embeddings"
mkdir -p $OUT_DIR
for batch_num in $batch_list; do
    JOBN="batch"$batch_num
    OUT_FILE=$OUT_DIR"/"$JOBN".npz"
    COMPLETE=0
    #ls $OUT_FILE
    if [[ -f $OUT_FILE ]]; then
    COMPLETE=1
    ((comp++))

    fi

    if [[ $COMPLETE -eq 0 ]]; then
    ((incomp++))
    # Script to be run
    SCRIPT="celeb_preprocess.sh $batch_num $batch_size"
    # Define job name
    OUTF=$LOGS"/"$JOBN".out"
    ERRF=$LOGS"/"$JOBN".err"
    # Assemble slurm order for this job
    ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
    # Print order
    echo $ORD
    # Submit order
    $ORD
    fi
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
