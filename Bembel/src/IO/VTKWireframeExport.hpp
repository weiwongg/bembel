// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_IO_VTKWIREFRAMEEXPORT_H_
#define BEMBEL_IO_VTKWIREFRAMEEXPORT_H_

namespace Bembel {

// This class provides the possibilty to generate a vtk-visualization.
class VTKWireframeExport {
 public:
  /**
   * \ingroup IO
   * \brief Provides export routines to the VTK file format.
   *
   * The constructor wants a geometetry and a refinement level. This choice is
   * deliberately not a mesh, since the visualization will often be on a finer
   * mesh then that of a computation.
   **/
  VTKWireframeExport(const Geometry& geo, int lvl, int lvl_x, int lvl_y) {
    init_VTKWireframeExport(geo, lvl, lvl_x, lvl_y);
  }
  inline void init_VTKWireframeExport(const Geometry& geo, int lvl, int lvl_x,
                                      int lvl_y) {
    msh.init_ClusterTree(geo, 1);
    num_ptc = geo.get_number_of_patches();
    std::cout << num_ptc << std::endl;
    lvl_ = lvl;
    lvl_x_ = lvl_x;
    lvl_y_ = lvl_y;

    return;
  }

  inline void writeToFile(const std::string& filename) {
    std::ofstream output;
    output.open(filename + ".vts.pvd");
    output << "<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"Collection\" version=\"0.1\">\n"
              "<Collection>\n";
    for (int iptc = 0; iptc < num_ptc; ++iptc) {
      output << "<DataSet part=\"";
      output << iptc;
      output << "\" file=\"";
      output << filename;
      output << ".vtu_";
      output << iptc;
      output << ".vtu\"/>\n";
    }
    output << "</Collection>\n"
              "</VTKFile>";
    output.close();
    for (int iptc = 0; iptc < num_ptc; ++iptc) {
      std::ofstream vtuoutput;
      vtuoutput.open(filename + ".vtu" + "_" + std::to_string(iptc) + ".vtu");
      vtuoutput << "<?xml version=\"1.0\"?>\n";
      vtuoutput << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">\n";
      vtuoutput << "<UnstructuredGrid>\n";
      vtuoutput << "<Piece NumberOfPoints= \"";
      vtuoutput << (pow(2, lvl_) + 1) * (pow(2, lvl_) + 1);
      vtuoutput << "\" NumberOfCells=\"";
      vtuoutput << (pow(2, lvl_x_) + 1) + (pow(2, lvl_y_) + 1);
      vtuoutput << "\">\n";
      vtuoutput << "<Points>\n";
      vtuoutput << "<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n";

      // output pts
        int n = pow(2, lvl_) + 1;
        // set up interpolation points in [0,nx]
        Eigen::VectorXd ret = (3.1415926) * Eigen::VectorXd::LinSpaced(n, 0., 1.0);
        ret = 0.5 * (1 - ret.array().cos());
        
      for (int j = 0; j < pow(2, lvl_) + 1; ++j) {
        for (int i = 0; i < pow(2, lvl_) + 1; ++i) {
          Eigen::Vector2d pt2d;
          pt2d << ret(i), ret(j);
          Eigen::Vector3d pt3d = msh.get_geometry()[iptc].eval(pt2d);
          vtuoutput << pt3d(0);
          vtuoutput << " ";
          vtuoutput << pt3d(1);
          vtuoutput << " ";
          vtuoutput << pt3d(2);
          vtuoutput << " ";
        }
      }
      vtuoutput << "</DataArray>\n";
      vtuoutput << "</Points>\n";
      vtuoutput << "<Cells>";
      vtuoutput << "<DataArray type=\"Int32\" Name=\"connectivity\" "
                   "format=\"ascii\">\n";
      int step_x = pow(2, lvl_) / pow(2, lvl_x_);
      int step_y = pow(2, lvl_) / pow(2, lvl_y_);
      for (int j = 0; j < pow(2, lvl_y_) + 1; ++j) {
        for (int i = 0; i < pow(2, lvl_) + 1; ++i) {
          vtuoutput << i + (j * step_y) * (pow(2, lvl_) + 1);
          vtuoutput << " ";
        }
        vtuoutput << "\n";
      }
      for (int i = 0; i < pow(2, lvl_x_) + 1; ++i) {
        for (int j = 0; j < pow(2, lvl_) + 1; ++j) {
          vtuoutput << i * step_x + j * (pow(2, lvl_) + 1);
          vtuoutput << " ";
        }
        vtuoutput << "\n";
      }
      vtuoutput << "</DataArray>\n";
      vtuoutput
          << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
      int offset = 0;
      for (int j = 0; j < pow(2, lvl_y_) + 1; ++j) {
        offset = offset + (pow(2, lvl_) + 1);
        vtuoutput << offset;
        vtuoutput << " ";
      }
      for (int i = 0; i < pow(2, lvl_x_) + 1; ++i) {
        offset = offset + (pow(2, lvl_) + 1);
        vtuoutput << offset;
        vtuoutput << " ";
      }
      vtuoutput << "\n";
      vtuoutput << "</DataArray>\n";
      vtuoutput
          << "<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\n";
      for (int j = 0; j < pow(2, lvl_y_) + 1; ++j) {
        vtuoutput << 4;
        vtuoutput << " ";
      }
      for (int i = 0; i < pow(2, lvl_x_) + 1; ++i) {
        vtuoutput << 4;
        vtuoutput << " ";
      }
      vtuoutput << "\n";
      vtuoutput << "</DataArray>\n";
      vtuoutput << "</Cells>\n";
      vtuoutput << "</Piece>\n";
      vtuoutput << "</UnstructuredGrid>\n";
      vtuoutput << "</VTKFile>";
      vtuoutput.close();
    }
    return;
  }

 private:
  ClusterTree msh;
  int lvl_;
  int lvl_x_;
  int lvl_y_;
  int num_ptc;
};

}  // namespace Bembel

#endif

