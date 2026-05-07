PID=$(pgrep -u $(whoami) -f runlbm.sh) #PID of runlbm.sh, check the user also to make sure I am checking my simulation
SRC_CSV="./Output"
SRC_PAR="./result_particle_scatter_binary"
DEST_CSV="./20260501_output_cubes_small_test"
DEST_PAR="./20260501_particle_cubes_small_test"

if [ -z "$PID" ]; then
    echo "Error: No simulation found for user $(whoami)."
    # Stop the script if nothing is running
elif [ $(echo $PID | wc -w) -gt 1 ]; then
    echo "Warning: Multiple simulations found for your user: $PID"
    echo "Please kill the old one or specify the PID manually."
else
    # Wait loop checking if simulation is still running
    echo "Monitoring PID $PID, will run when PID finishes"
    while kill -0 $PID 2>/dev/null; do
        sleep 60
    done
    # Move operation
    echo "Moving files to destination folders"

    # Ensure destination folder exists
    mkdir -p "$DEST_CSV"
    mkdir -p "$DEST_PAR"

    # Move all the files to the destination
    mv "$SRC_CSV"/* "$DEST_CSV/"
    mv "$SRC_PAR"/* "$DEST_PAR/"

    echo "Done. Output CSV moved to $DEST_CSV and particle binary moved to $DEST_PAR" 
fi