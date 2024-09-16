export def main [opt: string, test: string] {
  cd examples
  let filename = $opt ++ _ ++ $test
  cargo run --example $filename --release
    | $in ++ "\ne"
    | gnuplot -p -e $"
        set term pngcairo size 800,600;
        set output '($opt)/($test).png';
        set key noenhanced;
        plot '-' with points ps 1 title '($test) - ($opt)';
    "
}