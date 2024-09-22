export def all [] {
  for opt in [nsga2 spea2] {
    schaffer-n1 $opt
    binh-korn $opt
  };
}

export def schaffer-n1 [opt: string] {
  scatter-2d $opt "schaffer_n1"
}

export def binh-korn [opt: string] {
  scatter-2d $opt "binh_korn"
}

def scatter-2d [opt: string, test: string] {
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