#include <Bembel/IO>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  Bembel::Geometry geo("bunny.dat");
  const int refinement_level = atoi(argv[1]);
  const int p = geo.get_number_of_patches();
  const int n = 1 << refinement_level;
  const double h = 1. / double(n);
  const double scaling_factor = 1.;
  const Bembel::PatchVector &pv = geo.get_geometry();
  std::ofstream file;
  std::cout << "p: " << p << " n: " << n << " h: " << h << std::endl;
  file.open("geometry" + std::to_string(refinement_level) + ".bpd");
  file << refinement_level << std::endl << p << std::endl;
  for (auto i1 = 0; i1 < p; ++i1)
    for (auto i2 = 0; i2 <= n; ++i2)
      for (auto i3 = 0; i3 <= n; ++i3) {
        Eigen::Vector2d P2D;
        P2D << h * double(i3), h * double(i2);
        Eigen::Vector3d P3D = pv[i1].eval(P2D);
        file << std::setprecision(15) << i1 << " " << i2 << " " << i3 << " "
             << scaling_factor * P3D(0) << " " << scaling_factor * P3D(1) << " "
             << scaling_factor * P3D(2) << std::endl;
      }

  file.close();
  return 0;
}
