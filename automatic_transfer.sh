PID=$(pgrep -u $(whoami) -f runlbm.sh) #PID of runlbm.sh, check the user also to make sure I am checking my simulation
SRC="./Output"
DEST="./20260424_output_lobsmask"

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
    echo "Moving files to destination folder"

    # Ensure destination folder exists
    mkdir -p "$DEST"

    # Move all the files to the destination
    mv "$SRC"/* "$DEST/"

    echo "Done. File moved to $DEST"
fi