# ImplicitFeatureEncoding

This repository implements the method described in the following [paper](https://www.mdpi.com/2072-4292/15/1/61):
```
Zexin Yang, Qin Ye, Jantien Stoter, and Liangliang Nan. 
Enriching Point Clouds with Implicit Representations for 3D Classification and Segmentation.
Remote Sensing. 15(1), 61, 2023.
```

## Build
The current implementation depends on the [Point Cloud Library (PCL)](https://pointclouds.org).
Please install [PCL](https://pointclouds.org/downloads/#cross-platform) first.

To build ImplicitFeatureEncoding, you need [CMake](https://cmake.org/download/) (`>= 3.12`) and a compiler that supports `>= C++17`.
With CMake, ImplicitFeatureEncoding can be built on almost all platforms,
although so far we have only tested it on Linux (GCC >= 4.8, Clang >= 3.3).

There are many options to build ImplicitFeatureEncoding. Choose one of the following (not an exhaustive list):

- Option 1 (purely on the command line): Use CMake to generate Makefiles and then `make` (on Linux/macOS) or `nmake`(on
  Windows with Microsoft
  Visual Studio).
    - On Linux or macOS, you can simply
      ```
      $ cd path-to-root-dir-of-ImplicitFeatureEncoding
      $ mkdir Release
      $ cd Release
      $ cmake -DCMAKE_BUILD_TYPE=Release ..
      $ make
      ```
    - On Windows with Microsoft Visual Studio, use the `x64 Native Tools Command Prompt for VS XXXX` (**don't** use the
      x86 one), then
      ```
      $ cd path-to-root-dir-of-ImplicitFeatureEncoding
      $ mkdir Release
      $ cd Release
      $ cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
      $ nmake
      ```

- Option 2:
  Use any IDE that can directly handle CMakeLists files to open the `CMakeLists.txt` in the **root** directory
  of ImplicitFeatureEncoding.
  Then you should have obtained a usable project and just build it.
  I personally highly recommend using [CLion](https://www.jetbrains.com/clion/).
  For Windows users: your IDE must be set for `x64`.

- Option 3:
  Use CMake-Gui to generate project files for your IDE.
  Then load the project to your IDE and build it.
  For Windows users: your IDE must be set for `x64`.

Don't have any experience with C/C++ programming?
Have a look at <a href="https://github.com/LiangliangNan/Easy3D/blob/main/HowToBuild.md">Liangliang's step-by-step
tutorial</a>.

## Usage
The current implementation directly generates augmented point clouds with implicit features. 
These augmented point clouds can then be used as input for point-based neural networks.
The [main.cpp](./code/main.cpp) file demonstrates how to use the ImplicitFeatureEncoding class.
To encode implicit features, run the built executable like this:
```commandline
./ImplicitFeatureEncoding  <input_path>  <sample_sphere_ply_file>  <neighbor_radius>  <whether_to_fix_z_axis>  <output_path>
```
- `<input_path>` path to the input point cloud files (ply format)
- `<sample_sphere_ply_file>` path to the [sample sphere file](./test_data/sample_spheres) (ply format)
- `<neighbor_radius>` radius used for spherical neighborhood search
- `<whether_to_fix_z_axis>` set to 1 for real-world scene datasets (e.g.,
  [S3DIS](http://buildingparser.stanford.edu/dataset.html) and [SensatUrban](https://github.com/QingyongHu/SensatUrban),
  and set to 0 for synthetic datasets (e.g., [ModelNet40](https://modelnet.cs.princeton.edu/))
- `<output_path>` path where the resulting augmented point cloud files will be saved

## Citation
We kindly ask you to cite our paper if you use (part of) the code or ideas in your academic work:

```bibtex
@article{yang2022enriching,
  title={Enriching Point Clouds with Implicit Representations for 3D Classification and Segmentation},
  author={Yang, Zexin and Ye, Qin and Stoter, Jantien and Nan, Liangliang},
  journal={Remote Sensing},
  volume={15},
  number={1},
  pages={61},
  year={2022},
  publisher={MDPI}
}
```

---------

Please feel free to contact me at [zexinyang@gzpi.com.cn](zexinyang@tongji.edu.cn) with questions, comments, or suggestions ;-)

**_Zexin Yang_**

December 12, 2022
