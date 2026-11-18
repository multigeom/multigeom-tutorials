# Tests

The tests check that the notebooks and the figure scripts of this repository still run and still give the results stored in
`fixtures/`, and that the functions they use give known answers. They need the `multigeom` package installed (see the
repository README).

The estimators themselves, such as those of the multiscale Fisher information G(x), the Fisher information of the
responses J(x) and, from the next release, G(x) at a single noise level M_t(x), are tested in the package repository:
[multigeom/tests/unit](https://github.com/multigeom/multigeom/tree/main/tests/unit). Those that check what the notebooks
of this repository rely on are listed in the unit tests table below, under the name of their file in the package.

```bash
pytest                                                # everything; the tests that need data or checkpoints skip themselves without them
pytest -m "not slow and not gpu and not pretrained"   # what GitHub Actions runs on every pull request (CPU, no data, no checkpoints)
pytest tests/unit_tests                               # the unit tests only
```

Tests marked `slow` recompute results from the trained checkpoints (`runs/`) and the TinyImageNet data, or run a notebook
end to end. GitHub Actions does not run them; run them on your machine before regenerating a fixture. The fixtures are
regenerated with the scripts in `fixtures/` (`make_fig3_fixtures.py`, `make_fig4_de_fixtures.py`, `make_fig5_fixtures.py`,
`make_fig6_fixtures.py`, `make_per_scale_fixtures.py`, `make_stimulus_features_fixtures.py`,
`make_envelope_spectrum_fixtures.py`, `make_coding_spectrum_fixtures.py`).

The helpers are in `nb_helpers.py` (running the cells of a notebook, comparing two pictures, checking that the inputs of
the slow tests are there), `nb_fixtures.py` (the model-loading run of notebook 3, shared by its tests) and `conftest.py`
(environment settings).

## Writing a test

Give every test a docstring whose first paragraph says what it checks, starting with "Verify that": the tables below are
written from these paragraphs, the markers of the tests and their folder (`tests/` for regression tests,
`tests/unit_tests/` for unit tests) by `scripts/make_tests_readme.py`, one table per file, with the test names without
their `test_` prefix. The files are in the order of the notebooks they cover, given by a line `pytestmark =
pytest.mark.notebook(7)` after the imports of each file (files without it come last). Do not edit the tables by hand.
The script runs on every commit, once the hooks are installed (`pip install pre-commit`, then `pre-commit install` at
the repository root), and GitHub Actions fails when the tables are out of date (`python scripts/make_tests_readme.py
--check`). A test without such a docstring makes both fail. The hook needs a Python with `multigeom`: if you commit from
an editor that does not see your environment, tell it which one to use with `git config tests-readme.python
/path/to/python`.

## Regression tests

They check that the notebooks and the figure scripts still run, and still reproduce the figures and the numbers stored
in `fixtures/`. A figure is compared pixel by pixel with its reference picture: at most a mean absolute difference of
0.001, and at most 0.2% of the pixels off by more than 0.1. The figures of notebooks 3, 4, 5 and 7 to 10 also have a
`test_plot_comparison_detects_changes`, which changes the data and checks that the comparison then fails, so that it
cannot pass by accident.

<!-- regression-tests:start -->
52 tests: 43 run on GitHub Actions, 9 do not.

#### Notebooks 1, 2: [getting_started_notebooks](test_getting_started_notebooks.py)

Check that the getting-started notebooks 1 and 2 run end to end on a small fake dataset.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `notebook_1_builds_fits_saves_and_reloads_an_encoder` | Verify that notebook 1 runs on a tiny fake TinyImageNet: it builds and fits an encoder of 50 neurons, saves it, and the reloaded encoder predicts the same responses. | yes |
| `notebook_1_checks_the_contract_and_compares_fisher_with_the_teacher` | Verify that notebook 1 checks the encoder against the contract, and compares the leading eigenvectors of its Fisher information with those of the teacher (alignment of the leading k eigenvectors at k = 1, 5 and 10, between 0 and 1). | yes |
| `notebook_2_trains_a_diffusion_model_and_computes_g` | Verify that notebook 2 trains a small diffusion model and computes finite values of the multiscale Fisher information G(x) and the Fisher information of the responses J(x) of the expected shapes. | yes |

#### Notebook 3: [inception_leading_eigenvectors](test_inception_leading_eigenvectors.py)

Check that notebook 3 still reproduces the leading eigenvectors of the multiscale Fisher information G(x) and of the Fisher information of the responses J(x) for the Inception encoder (paper Fig. 3D-F).

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored data of Fig. 3D-F (notebook 3) is complete: receptive-field parameters of the V1 and V4 readouts, the frog, and unit-norm eigenvectors of the multiscale Fisher information G(x) and the Fisher information of the responses J(x). | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 3 draws Fig. 3D-F like the reference picture. | yes |
| `plot_comparison_detects_changes` | Verify that, with the four stored leading eigenvectors of the multiscale Fisher information G(x) and the Fisher information of the responses J(x) of V1 drawn in reverse order (the 4th in place of the 1st), Fig. 3D-F differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that notebook 3 recomputed from the checkpoints gives the stored eigenvectors: the leading 4 of the Fisher information of the responses J(x) with absolute cosine \|cos\| > 0.99, the leading one of the multiscale Fisher information G(x) with \|cos\| > 0.9 (G(x) is a Monte-Carlo estimate). | no (slow) |

#### Notebook 3: [resnet50_leading_eigenvectors](test_resnet50_leading_eigenvectors.py)

Check that notebook 3 still reproduces the leading eigenvectors of the multiscale Fisher information G(x) and of the Fisher information of the responses J(x) for the ResNet-50 encoder (paper Fig. 4B-C).

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored ResNet-50 eigenvectors of Fig. 4B-C (notebook 3) have the expected shape and unit norm. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 3 draws Fig. 4B-C like the reference picture. | yes |
| `plot_comparison_detects_changes` | Verify that, with the ResNet-50 eigenvectors of the multiscale Fisher information G(x) and the Fisher information of the responses J(x) of V1 replaced by those of V4, Fig. 4B-C differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that the ResNet-50 eigenvectors recomputed from the checkpoints agree with the stored ones: the leading 4 of the Fisher information of the responses J(x) with absolute cosine \|cos\| > 0.99, the leading one of the multiscale Fisher information G(x) with \|cos\| > 0.9. | no (slow) |

#### Notebook 4: [fig4_de_subspace_alignment](test_fig4_de_subspace_alignment.py)

Check that notebook 4 still reproduces how well the leading eigenvectors of different encoders and areas align (paper Fig. 4D-E).

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_has_the_four_comparisons_of_both_metrics` | Verify that the stored alignment metrics of Fig. 4D-E (notebook 4) have the four comparisons of the figure, for the multiscale Fisher information G(x) and the Fisher information of the responses J(x). | yes |
| `alignment_is_between_zero_and_one` | Verify that every stored alignment curve has 50 values, all between 0 and 1. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot of notebook 4 draws Fig. 4D-E like the reference picture. | yes |
| `plot_comparison_detects_changes` | Verify that, with the mean alignment curve of one comparison (Inception V1 against V4, the multiscale Fisher information G(x)) set to 0.9 for every number of eigenvectors k, Fig. 4D-E differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `fixture_info_lists_how_the_fixture_was_made` | Verify that `fixture_info.json` records the estimator settings, the images and the diffusion checkpoint of each configuration. | yes |
| `fixture_info_agrees_with_the_stored_metrics` | Verify that the configurations, images and number of eigenvectors in `fixture_info.json` are those of the stored metrics. | yes |
| `fixture_parameters_match_the_current_settings_of_the_repository` | Verify that the fixture was made with the current estimator config and plot defaults, so it still describes this code. | yes |

#### Notebook 5: [fig5_local_metric_field](test_fig5_local_metric_field.py)

Check that the stored local metric field of notebook 5 is valid and that its plot still reproduces paper Fig. 5.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_matrices_are_valid_metrics` | Verify that the stored 2 x 2 metrics of Fig. 5 (notebook 5) are symmetric and positive semi-definite, on the grid −2 to 2 around a 64 x 64 image. | yes |
| `fixture_info_lists_how_the_fixture_was_made` | Verify that `fixture_info.json` records the settings of each part (samples, grid size, step rule) and the diffusion checkpoints. | yes |
| `fixture_info_agrees_with_the_stored_matrices` | Verify that the grid size, metrics and areas in `fixture_info.json` are those of the stored matrices. | yes |
| `fixture_parameters_match_the_current_settings_of_the_repository` | Verify that the fixture was made with the current estimator config, job defaults (image, steps) and plot defaults. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot of notebook 5 draws Fig. 5 like the reference picture. | yes |
| `plot_comparison_detects_changes` | Verify that, with the ResNet-50 metrics of the multiscale Fisher information G(x) for V1 multiplied by 4 and given an off-diagonal term of 0.5, Fig. 5 differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |

#### Notebook 5: [local_metric_field_notebook](test_local_metric_field_notebook.py)

Check that the cells of notebook 5 set up, run and plot a local metric field job.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `the_notebook_exists_and_has_the_cells_the_tests_use` | Verify that notebook 5 has the code cells that the other tests run. | yes |
| `parameter_cell_gives_the_documented_job_and_its_cost` | Verify that the parameter cell of notebook 5 sets the documented job (image 64, a 5 x 5 grid, the step rules of the multiscale Fisher information G and the Fisher information of the responses J, Inception against ResNet-50) and prints its cost (102 G(x) estimates and 102 J(x) evaluations). | yes |
| `plot_cell_draws_the_reference_figure_from_stored_results` | Verify that the plot cell of notebook 5 draws Fig. 5 from stored results like the reference picture. | yes |
| `plot_cell_says_so_when_there_are_no_results` | Verify that, without results, the plot cell says "Nothing to plot yet" instead of failing. | yes |
| `notebook_runs_a_small_job_end_to_end` | Verify that notebook 5 runs a small job (the Fisher information of the responses J(x) only, V1, a 3 x 3 grid) from launch to plot in a scratch copy of the repository, and gives symmetric metrics. | no (slow) |

#### Notebook 6: [bures_wasserstein_notebook](test_bures_wasserstein_notebook.py)

Check that notebook 6 still computes and plots the Bures-Wasserstein distances between the geometries of encoders and areas (paper Fig. 6).

| Test | What it checks | GitHub Actions |
|---|---|---|
| `the_notebook_has_the_cells_the_tests_use` | Verify that notebook 6 has the code cells that the other tests run. | yes |
| `fixture_info_agrees_with_its_source_and_the_repository` | Verify that the stored metrics of Fig. 6 come from the recorded source file (MD5), give the paper's ordering of the comparisons, and match the current estimator config and plot function. | yes |
| `ordering_cell_reports_the_papers_order` | Verify that the ordering cell of notebook 6 finds the paper's ordering of the comparisons, for the multiscale Fisher information G(x) and the Fisher information of the responses J(x). | yes |
| `plot_cell_draws_the_reference_figure` | Verify that the plot cell of notebook 6 draws Fig. 6 like the reference picture. | yes |
| `plot_and_ordering_cells_say_so_without_results` | Verify that, without results, the cells say "Nothing to check yet" and "Nothing to plot yet" instead of failing. | yes |
| `notebook_runs_stage5_and_plots_in_a_scratch_copy` | Verify that notebook 6 computes the metrics from the per-image results of notebook 4 and plots them, in a scratch copy, without touching the real results. | no (slow) |

#### Notebook 7: [eigenvectors_per_scale](test_eigenvectors_per_scale.py)

Check that notebook 7 still computes and plots the leading eigenvectors of the multiscale Fisher information G(x) at a single noise level, M_t(x), for the frog.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored data of notebook 7 (the frog) has 24 increasing noise levels and, for V1 and V4, 4 positive decreasing eigenvalues and unit-norm eigenvectors at each level. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 7 draws the figure of the frog like the reference picture, and saves it. | yes |
| `plot_comparison_detects_changes` | Verify that, with the four leading eigenvectors of V1's multiscale Fisher information at a single noise level, M_t(x), drawn in reverse order (the 4th in place of the 1st) at every noise level, the figure of the frog differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that notebook 7 recomputed from the checkpoints gives the stored noise levels, and at the coarse levels (log noise variance τ ≥ 3) the stored leading eigenvector (absolute cosine \|cos\| > 0.9) and leading eigenvalue (within 30%). | no (slow) |

#### Notebook 8: [stimulus_features_per_scale](test_stimulus_features_per_scale.py)

Check that notebook 8 still computes and plots the stimulus features at a single noise level, M_t^env(x), for the frog.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored stimulus features of notebook 8 (the frog) have the noise levels of notebook 7's fixture and 4 positive decreasing eigenvalues and unit-norm eigenvectors at each level. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 8 draws the figure of the frog (stimulus features, then V1 and V4 from notebook 7's fixture) like the reference picture, and saves it. | yes |
| `plot_comparison_detects_changes` | Verify that, with the four leading eigenvectors of the stimulus features at a single noise level (what stays uncertain about the image given its noisy version), M_t^env(x), drawn in reverse order (the 4th in place of the 1st) at every noise level, the figure of the frog differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that the stimulus features of the frog recomputed from the V1 checkpoint have the stored settings and noise levels, and at the coarse levels (log noise variance τ ≥ 3) the stored leading eigenvector (absolute cosine \|cos\| > 0.9) and leading eigenvalue (within 30%). | no (slow) |

#### Notebook 9: [envelope_spectrum](test_envelope_spectrum.py)

Check that notebook 9 still computes and plots the envelope spectrum, the error of the denoiser without the responses at each noise level.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored envelope spectrum of notebook 9 has 31 increasing noise levels, a positive curve for each of the 200 test images and both models, the variances of the training images largest first, and a Gaussian bound that decreases from below the number of pixels P. | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 9 draws the envelope spectrum like the reference picture, and saves it. | yes |
| `plot_comparison_detects_changes` | Verify that, with the V1 model's envelope spectrum (the denoising error without the responses at each noise level), ρ^env(τ), multiplied by 1.5 at every noise level, the envelope spectrum figure differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that the spectrum recomputed from the V1 and V4 checkpoints has the stored noise levels and mean curves (within 2%), and the training images give the stored variances and Gaussian bound. | no (slow) |

#### Notebook 10: [coding_spectrum](test_coding_spectrum.py)

Check that notebook 10 still computes and plots the coding spectrum, what the responses take off the error of the denoiser at each noise level, and the information it adds up to.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `fixture_is_wellformed` | Verify that the stored coding spectrum of notebook 10 has 31 increasing noise levels and, for V1 and V4 and each of the 200 test images, the four terms of the curve at each level (positive errors without and with the responses, non-negative changes made by the responses). | yes |
| `notebook_plot_reproduces_reference` | Verify that the plot cell of notebook 10 draws the coding spectrum like the reference picture, and saves it. | yes |
| `plot_comparison_detects_changes` | Verify that, with V4's coding spectrum (what the responses take off the denoising error at each noise level), ρ(τ), doubled at every noise level (its error with the responses moved twice as far from the error without them), the coding spectrum figure differs from the reference picture beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean \|difference\| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that test cannot pass by accident. | yes |
| `recompute_matches_fixture` | Verify that the coding spectrum recomputed from the V1 and V4 checkpoints (about 7 min on a GPU) has the stored noise levels, the stored mean errors without and with the responses (within 2%) and the mutual information between the responses and the image, ½∫ρ dτ (within 1 nat). | no (slow) |

#### [figures](test_figures.py)

Check that every figure and table script of `figures/` still writes its outputs.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `figure_scripts_write_outputs` | Verify that every figure and table script of `figures/` writes its files from a small synthetic `runs/` tree, and skips what is missing (the ON-OFF encoder). | no (slow) |
<!-- regression-tests:end -->

## Unit tests

They check one function on inputs whose answer is known, without stored references, data or checkpoints. Those that
need a function newer than the `multigeom` that GitHub Actions installs (the package's `main` branch, or `develop`
when testing `develop`) are skipped there until it has it.

<!-- unit-tests:start -->
15 tests: 15 run on GitHub Actions (6 of them skipped while what they need is missing). Plus 5 of the multigeom package.

#### Notebooks 0, 1, 2, 3, 4, 7, 8: [eigenpairs](unit_tests/test_eigenpairs.py)

Check that every eigenvector the notebooks plot is a true eigenvector of its matrix, with the right eigenvalue.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `eigenvectors_are_eigenvectors` | Verify that every eigenvector that the notebooks plot is an eigenvector of its matrix, with the right eigenvalue: for the Fisher information of the responses J(x) (all its pairs), the multiscale Fisher information G(x), G(x) at each noise level, M_t(x), and the stimulus features at each noise level, M_t^env(x) (the top 10 pairs, at every noise level), each pair of eigenvalue and eigenvector (λ_i, v_i) has a relative residual ‖M v_i − λ_i v_i‖ / ‖M‖ below 1e-5 and a unit eigenvector. M v_i is computed as Dᵀ(D v_i) from the rows D of the same samples that the estimator drew, without building M. | yes\* |
| `the_check_fails_on_mismatched_eigenpairs` | Verify that eigenpairs that are not those of DᵀD fail the residual check, so that the test above cannot pass by accident: the eigenvalues paired with the next eigenvector (a shifted index), eigenvectors whose 8 x 8 images are transposed (a wrong reshape), and an eigenvector scaled by 2 (not a unit vector). | yes |

\* Some of its cases are skipped while what they need is missing (needs a multigeom with compute_g_per_scale; needs a multigeom with compute_env_per_scale).

#### Notebook 7: [compute_g_per_scale](unit_tests/test_compute_g_per_scale.py)

Check that the estimator of the multiscale Fisher information G(x) at a single noise level, M_t(x), gives the known answer on mock models.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `m_t_is_zero_when_the_denoiser_ignores_the_response` | Verify that, with a mock denoiser that ignores the response r, both responses give the same noise prediction, so `compute_g_per_scale` returns a multiscale Fisher information at a single noise level M_t(x) = 0: every eigenvalue is exactly 0 at every noise level. | yes\* |
| `m_t_is_not_zero_when_the_denoiser_reads_the_response` | Verify that the same mock plus a term that reads r gives a positive leading eigenvalue at every noise level, so the zero above is not vacuous. | yes\* |
| `every_difference_is_zero_when_the_responses_are_identical` | Verify that, with no observation noise (response noise σ = 0) the two responses of each sample are identical, r = r′, so every row of the matrix D that the conditional-pair method draws, the difference of the noise predictions with the two responses, ε(x_t, t, r) − ε(x_t, t, r′), is exactly zero at every noise level, even with a mock denoiser that reads r. | yes |
| `differences_are_not_zero_when_the_responses_differ` | Verify that, with the default response noise σ (0.22) the responses differ, and the same mock gives a non-zero row for every sample at every noise level. | yes |
| `trace_of_m_t_is_the_sum_of_squared_differences_over_2nt` | Verify that, at every noise level, the sum of all the eigenvalues of the estimated multiscale Fisher information at that noise level is tr M̂_t = Σ_s \|δ_s\|² / (2 N t), where δ_s = x̂(x_t, r) − x̂(x_t, r′) is the difference of the two denoised images of sample s (Tweedie's formula, in image units, from the calls recorded by the mock denoiser), N the number of samples and t the noise variance. | yes\* |
| `explicit_matrix_is_symmetric_and_positive_semidefinite` | Verify that, at every noise level, the whole estimated multiscale Fisher information at that noise level, a P x P matrix over the P pixels, M̂_t = Σ_s δ_s δ_sᵀ / (2 N t), built from the differences of the denoised images that the estimator drew, is symmetric and positive semi-definite within numerical tolerance (no eigenvalue below −1e-10 times the largest; its zero eigenvalues come out around −1e-16); so is the matrix rebuilt from the eigenpairs that `compute_g_per_scale` returns, and the two agree (relative difference below 1e-5). | yes\* |
| `the_curve_equals_t_times_the_trace_of_the_fisher_information` | Verify that the trace of the estimated multiscale Fisher information at each noise level, trace(M̂_t), equals the noise variance t times the trace of the Fisher information of the responses about the noisy image, t · trace(⟨J_t⟩), at five noise levels from log t = −6.1 to 8.2 (agreement within 1.3%, tolerance 10%). It uses a linear-Gaussian model whose posterior means and Fisher information are known in closed form, and draws the responses with the spread they have given the noisy image; with the estimator's own draws, given the denoised image, the curve is lower (8% to 99% of it in this model, the coarser the lower), which the test also catches. | yes\* |

\* Skipped while it needs a multigeom with compute_g_per_scale.

#### Notebooks 7, 8, 9, 10: [per_scale_helpers](unit_tests/test_per_scale_helpers.py)

Check the helpers shared by notebooks 7 to 10 (sign alignment, Gaussian bound, integration) on inputs whose answer is known.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `align_signs_flips_only_signs_to_agree_with_coarser_levels` | Verify that `align_signs` (`figures/per_scale.py`, notebooks 7 and 8) only flips signs, makes each eigenvector agree with the same one at the next coarser level, and gives the coarsest a positive sum. | yes |
| `compute_gaussian_env_on_known_variances` | Verify that `compute_gaussian_env` (`figures/per_scale.py`, notebook 9), the envelope spectrum of a Gaussian prior, gives the sum over the directions of their variance c_i over c_i plus the noise variance t, Σ_i c_i / (c_i + t), on known variances: the number of directions that vary at tiny noise, ½ per unit-variance direction at t = 1, and the total variance over t, tr C / t, at huge noise. | yes |
| `compute_information_on_known_curves` | Verify that `compute_information` (`figures/per_scale.py`, notebook 10) gives half the area under the curve of each image (trapezoids over log t), the mean over the images and its standard error, on two piecewise-linear curves of known area. | yes |
| `compute_normalized_on_known_values` | Verify that `compute_normalized` (`figures/per_scale.py`, notebooks 7, 8 and 10) divides the values by the largest value of the reference: a curve by its own maximum, and eigenvalues by the largest leading eigenvalue, so that the second ones keep their size relative to it. | yes |

#### Notebook 8: [env, in the multigeom package](https://github.com/multigeom/multigeom/blob/main/tests/unit/test_env.py)

Check that the estimator of the stimulus features at a single noise level, M_t^env(x), gives the known answer for a Gaussian prior and in two exact limits.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `eigenvalues_are_c_over_c_plus_t_not_squared` | Verify that, for a Gaussian prior with variances c_i whose answer is known, the four leading eigenvalues of the stimulus features at a single noise level, M_t^env(x), equal c_i / (c_i + t), with t the noise variance, at five noise levels (within 10%), and not their squares. | in the package's checks (from its next release) |
| `each_eigenvector_matches_its_own_prior_direction` | Verify that, for a Gaussian prior whose answer is known, each leading eigenvector of the stimulus features at a single noise level, M_t^env(x), points along its own direction of image variation, in the right order (absolute cosine \|cos\| above 0.95), at every noise level where neighbouring eigenvalues differ by at least 20%. | in the package's checks (from its next release) |
| `a_posterior_mean_that_ignores_x_t_gives_zero_rows` | Verify that, with a denoiser whose estimate of the clean image ignores the noisy image x_t, every row of probes that the estimator of the stimulus features draws is exactly zero, at each noise level: the stimulus features are then zero. | in the package's checks (from its next release) |
| `the_identity_gives_rows_equal_to_their_probes` | Verify that, with a denoiser that returns the noisy image unchanged, every row that the estimator of the stimulus features draws equals the random probe that produced it, at each noise level: the stimulus features are then the identity matrix. | in the package's checks (from its next release) |
| `the_identity_has_singular_values_near_one_with_many_probes` | Verify that, with the same denoiser that returns the noisy image unchanged and many probes, the eigenvalues that the estimator of the stimulus features returns are all close to 1, as for the identity matrix. | in the package's checks (from its next release) |

#### [figure_style](unit_tests/test_figure_style.py)

Check the helpers of `figures/style.py` that set the plotting backend and read the scores of the encoders.

| Test | What it checks | GitHub Actions |
|---|---|---|
| `style_keeps_the_notebook_backend_in_a_jupyter_kernel` | Verify that importing `figures.style` in a Jupyter kernel keeps the notebook backend, so figures show as pictures (it switches to Agg only in scripts). | yes |
| `encoder_checkpoint_gives_the_scores_in_both_layouts` | Verify that `encoder_checkpoint` reads the test scores of an encoder checkpoint in both layouts (older paper checkpoints and `NeuralEncoder.save`), and returns None when there is none. | yes |
<!-- unit-tests:end -->
