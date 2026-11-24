# PET Virtual Laboratory v14

## v14 — independent source diameters, asymmetric activity and a more teachable resolution comparison

This update corrects the reconstruction display error, makes the sinogram source-position controls visibly interactive, and refines detector PSF modelling to act in projection space.

### Fixes and sinogram controls
- Corrected the Matplotlib call to `set_yticks([])` in the two-source ROI plot. The previous empty `set_yticks()` call caused the reported `Axis.set_ticks()` TypeError.
- The independent point-source controls now update two linked views: source position and selected projection line in object space, and the sinusoidal detector-offset trajectory with a marker at the selected angle.
- The x/y/angle controls are a separate geometry demonstrator; they do not move the two-source phantom. The two-source phantom has its own diameter, separation and activity controls.
- Added adjustable scatter and random fractions to the sinogram simulation.
- Explained the central-source and symmetry cases: an object-centred point has s=0 at every angle, while an off-centre point traces a sinusoid. Symmetric source trajectories can cross at particular angles.

### Detector PSF modelling refinement
- Replaced the previous 2-D object-image Gaussian blur approximation with a Gaussian response across detector-offset bins independently within each projection view.
- The measured data are generated using the detector response operator G applied to the ideal projection matrix A (G·A).
- PSF-off reconstruction uses A and deliberately ignores detector-bin blur; PSF-aware reconstruction uses G·A. Both are compared using exactly the same prompt data.
- This is a simplified stationary projection-space detector response for teaching; real PET system response is spatially variant and scanner-specific.

### Updated two-source experiment
- Independent source diameters are now supported throughout the phantom, sinogram, reconstruction and ROI analysis. The left and right ROIs are sized separately.
- Default left source: 3 mm diameter and 4 × background activity.
- Default right source: 5 mm diameter and 8 × background activity.
- Default centre-to-centre separation: 8.0 mm, leaving a 4.0 mm physical edge-to-edge gap.
- Independent diameter, activity and centre-separation controls are available in both Sinogram and Reconstruction modules.
- The minimum centre spacing follows the sum of the two radii plus a small gap, preventing the two source regions from overlapping.
- The sinogram path diagram now draws a projected-width band around each centre trajectory; the band width reflects source diameter and its label shows both diameter and activity.
- Quantitative ROIs follow the corresponding source diameters, and left/right contrast recovery is calculated separately against the same range-blurred target.
- Source profiles show the physical extent of each source independently. The default pair is intended to demonstrate unequal-size partial-volume recovery and neighbouring-source spill-in at a realistic teaching scale, not to promise that PSF modelling can recover arbitrarily close sources.
- Added a controlled image-domain PSF experiment in the Reconstruction module: the positron-range-blurred two-source target is convolved with a known detector Gaussian PSF, then restored with Richardson–Lucy iterations. This isolates detector-PSF restoration and displays before/after left/right recovery and central-gap contrast. It is deliberately noise-free to teach the principle; the paired MLEM/OSEM projection-space PSF experiment remains separate and more physically complete.
- The source sliders use new widget keys so older Streamlit session-state values from the previous equal-diameter experiment do not silently override the new defaults.

### Validation and limitations
- Python syntax compilation passed.
- Targeted numerical tests confirmed source/background/gap ROIs are populated and do not overlap for the default geometry, independent point-source position/angle changes alter detector offset, detector PSF operator rows are normalised, and PSF-aware versus PSF-unaware reconstructions differ on the same simulated prompts.
- Targeted helper tests confirmed distinct source-mask sizes (9 pixels for the 3-mm source and 21 pixels for the 5-mm source on the 1-mm teaching grid), no ROI overlap, correct left/right nominal activity and 100% ROI recovery when the supplied image equals the target. The sinogram trajectory plot also renders with different source-size bands.
- For the default F-18, 8-mm separation and 4.5-mm detector-FWHM image-domain teaching demonstration, the deterministic blurred profile has one local peak while the known-PSF Richardson–Lucy restoration has two. This is a noise-free explanatory example, not proof that clinical PSF reconstruction can always resolve a pair at that spacing.
- The application remains a 2-D educational parallel-beam model with stationary Gaussian detector-bin response and Gaussian-equivalent positron-range blur. It is not a clinical reconstruction or validated resolution test.
- A live Streamlit browser launch was not performed in the build environment.

---

## v11 — close two-source resolution and quantification experiment

This release adds a dedicated pair of nearby hot sources to both the Sinogram and Reconstruction modules. The default experiment uses two symmetric 4 mm hot disks, 8 mm centre-to-centre apart, at 8:1 activity-to-background contrast. Students can vary source diameter, separation and activity. The separation control maintains a small non-overlapping gap so that the two source ROIs and central gap ROI remain independently interpretable.

### Sinogram module
- The two-source phantom is now the default example.
- Shows the same source phantom with an idealised detector response (detector PSF = 0) and with the selected detector PSF. The isotope-dependent positron-range kernel, scatter/random fractions and count scale are retained in both comparisons; only detector PSF changes.
- Adds the expected sinusoidal centre trajectories of both sources across projection angles. The two ideal paths cross at 90 degrees for the symmetric horizontal arrangement.
- Explains that detector PSF broadens the measured projection response; PSF modelling is applied during reconstruction, not by changing the acquired sinogram.

### Reconstruction module
- Uses the same two-source geometry and adjustable source separation, diameter and uptake.
- Displays independent left/right source ROIs and a central gap ROI on the reconstructed image, plus a central line profile.
- Reports background-normalised source means, background-subtracted contrast recovery against the positron-range-blurred target, central gap activity and gap-to-hot contrast ratio.
- The paired PSF experiment reconstructs the exact same prompt events with PSF modelling OFF and ON, then compares source recovery and inter-source spill-in in a table.
- The two-source teaching notes explain partial-volume effects, apparent merging, spill-in/spill-out, PSF-aware reconstruction benefits and possible noise/edge artefacts.

### Interpretation and limitations
The source recovery is calculated from background-subtracted contrast:

`RC = 100 × (mean reconstructed source − mean reconstructed background) / (mean target source − mean target background)`

The gap-to-hot contrast ratio is an illustrative cross-talk/separability metric for this digital phantom, not a standardised clinical lesion-detectability or scanner-resolution metric. The experiment is a 2-D parallel-beam teaching simulation with a stationary Gaussian detector PSF and a Gaussian-equivalent positron-range blur. It is not a clinical reconstruction engine, measured system PSF, NEMA resolution test or validated partial-volume correction.

### Validation
- Python syntax compilation passed.
- Targeted numerical checks confirmed both source ROIs and the central gap ROI are populated and non-overlapping for the default geometry.
- PSF OFF/ON reconstructions ran on identical prompt data. In the default F-18 teaching test, PSF modelling increased mean source contrast recovery from approximately 42% to 51% and reduced the central gap-to-hot contrast ratio from approximately 41% to 38%. These are illustrative results for that test seed and parameter set, not general performance claims.
- A live Streamlit browser launch was not performed in the build environment because Streamlit is not installed there.

---

# PET Virtual Laboratory v10

This patch fixes a further reconstruction-display `NameError`: the original activity phantom is stored as `truth_image`, so the original-phantom panel now uses that same defined variable rather than the undefined `source_truth` name. No reconstruction maths or QC calculations were changed in this patch.

# PET Virtual Laboratory v9

This release fixes the MLEM/OSEM convergence-study `NameError` by explicitly unpacking and retaining the positron-range-blurred comparison target separately from the original source phantom. Reconstruction scaling, difference maps and convergence metrics now consistently use the blurred target; the original phantom remains available for the source profile.

# PET Virtual Laboratory v8

Interactive Streamlit teaching application covering positron physics, PET coincidence detection, sinograms, FBP/MLEM/OSEM reconstruction, PSF resolution modelling, TOF concepts, partial-volume effects (PVE), and PET phantom/QC exercises.

## Run locally on macOS

Open Terminal and change to the folder containing `app.py` and `requirements.txt` first. For example:

```bash
cd "/Users/your-user/path/to/PET"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `.venv` already exists in this project folder, activate it rather than creating it again:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## v8 changes

- Added a resolution-path diagram with arrows showing where positron range, detector/system PSF, PSF-aware reconstruction and PVE enter the imaging chain. TOF is explained separately because it adds localisation along the LOR rather than removing physical blur.
- Added an interactive PVE demonstration. Students change hot-structure diameter and see the true disk, blurred disk, central profile, mean ROI recovery and recovery-versus-diameter curve.
- Clarified PSF modelling: the simulated measurements contain detector blur in both cases; the PSF switch controls whether the iterative reconstruction includes that detector response in its forward system matrix.
- Improved the paired PSF comparison so both images use the same display scale and prompt data, background-normalised quantitative comparisons, ROI metrics and a common central line profile.
- Added a reconstruction convergence study comparing MLEM (one subset) and OSEM with selectable subset counts using the same noisy prompt dataset.
- Added curves and a table for background CV, background SNR, hot contrast recovery, hot-to-background CNR, normalised RMSE and full-data Poisson log-likelihood.
- The convergence x-axis uses complete projection-data passes (nominal iterations). Each OSEM pass processes all projection angles through sequential subsets; subset-update count is shown separately. This avoids implying that `iterations × subsets` is directly equivalent to the number of full-data MLEM iterations.
- Added interpretation guidance explaining that higher contrast recovery or likelihood does not necessarily mean better task performance when noise rises, and that there is no universal optimum iteration/subset setting.
- Confirmed numerically that OSEM with one subset gives the same update as MLEM in this implementation.

## Validation and limitations

The Python source passes `py_compile`. Numerical helper tests cover MLEM/OSEM history generation, ROI metrics, the resolution-path diagram, and exact equivalence of MLEM with one-subset OSEM. In a 64 × 64 teaching test with detector blur in the simulated data, the paired PSF-aware reconstruction produced measurably different image recovery/noise metrics from PSF-unaware reconstruction.

The full Streamlit browser application was not launched in the build environment because Streamlit is not installed there. Install dependencies using the commands above before launching locally.

This remains an educational simulation, not clinical reconstruction or scanner QC software. The projector is a simplified 2-D parallel-beam model; the detector PSF is stationary and Gaussian; scatter/randoms and isotope-dependent positron-range kernels are simplified; TOF is taught and visualised but is not binned into the MLEM/OSEM system matrix. PVE is demonstrated using an idealised disk and Gaussian blur, not a validated partial-volume correction method. Formal NEMA performance testing and ACR accreditation must follow the current standard/accreditation instructions, phantom geometry, acquisition, reconstruction and analysis methods. Use commissioning baselines, manufacturer specifications, applicable regulation/accreditation requirements and physicist-approved local action limits for real scanner assessment.

### v14 interpretation note
The default pair is deliberately asymmetric: a 3-mm, 4× hot source next to a 5-mm, 8× hot source, with 8-mm centre spacing. This makes the size-dependent recovery and cross-talk visible without placing the sources unrealistically close. PSF-aware reconstruction may improve recovery of the smaller source if the assumed system response matches the data, but it cannot guarantee resolution of a pair that is below the information limit. Results depend on count statistics, positron range, detector FWHM, reconstruction algorithm, iterations/subsets, voxel size and model mismatch.
