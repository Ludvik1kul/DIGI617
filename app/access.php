<?php
if (isset($_GET['text'])) {
    $input = escapeshellarg($_GET['text']);  // safely escape input
    $command = "python3 model.py " . $input; // run Python
    $output = shell_exec($command);          // capture result
    echo $output;                            // return it
} else {
    echo "No input provided";
}
?>