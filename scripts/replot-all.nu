use replot.nu main
export def main [] {
  const opts = [nsga2 spea2]
  const tests = [binh_korn schaffer_n1]
  $opts
    | each { |o|
      $tests | each { |t| replot $o $t }
    }
  null
}