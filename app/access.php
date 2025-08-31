<?php
if (isset($_GET['text'])) {
    $input = escapeshellarg($_GET['text']);  // safely escape input
    $output = shell_exec("python3 /app/model.py " . $input);
    echo $output;                            // return it
} else {
    echo "No input provided";
}
?>