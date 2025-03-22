% scale(1000) import("u_side_femur_2023.stl");

// Sketch PureShapes 3.0
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 3.000000;
translate([0, 0, -thickness]) {
  translate([-15.000000, -96.500000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([30.000000, 96.500000, thickness]);
    }
  }
}
}
