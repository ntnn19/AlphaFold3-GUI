# Changelog

> Fork of [Hanziwww/AlphaFold3-GUI](https://github.com/Hanziwww/AlphaFold3-GUI)

## [2.0.0] - 2024-02-12

### Added
- Advanced 3D visualization controls
  - Multiple visualization styles (cartoon, stick, line, sphere)
  - Different coloring schemes (confidence, chain, secondary structure, rainbow, custom)
  - Residue selection and highlighting functionality
  - Chain sequence viewer in sidebar
- Export capabilities
  - Added support for multiple 3D file formats (CIF, PDB, STL)
  - New CAD/3D printing export functionality with customizable resolution
  - Improved mesh generation for 3D models
- Enhanced error handling
  - Robust CSV parsing with improved error messages
  - Better handling of different PAE data formats
  - Comprehensive logging system

### Changed
- Improved UI/UX
  - Added sidebar navigation
  - Reorganized visualization controls
  - Enhanced layout with better column ratios
  - Added interactive sequence selection
- Enhanced visualization performance
  - Optimized memory usage for large structures
  - Improved rendering efficiency
- Updated documentation and code structure
  - Better function organization
  - Improved code comments
  - Added type hints and error handling

### Fixed
- Various bug fixes in PAE matrix visualization
- Improved handling of missing or malformed data
- Fixed issues with chain boundary visualization
- Resolved memory leaks in 3D visualization

## [1.0.0] - Original Release

- Initial release with basic visualization features
- Basic structure visualization
- PAE matrix display
- Confidence metrics visualization
