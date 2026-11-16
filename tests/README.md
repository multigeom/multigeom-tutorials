# Tests

The tests check that the notebooks and the figure scripts of this repository still run and still give the results stored in
`fixtures/`, and that the functions they use give known answers. They need the `multigeom` package installed (see the
repository README).

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

## Regression tests

52 tests, 43 of which run on GitHub Actions (the 9 marked `slow` do not). They check that the notebooks and the figure
scripts still run, and still reproduce the figures and the numbers stored in `fixtures/`. A figure is compared pixel by
pixel with its reference picture: at most a mean absolute difference of 0.001, and at most 0.2% of the pixels off by
more than 0.1. The figures of notebooks 3, 4, 5 and 7 to 10 also have a `test_plot_comparison_detects_changes`, which
changes the data and checks that the comparison then fails, so that it cannot pass by accident.

| File | Test | What it checks | GitHub Actions |
|---|---|---|---|
| [`test_inception_leading_eigenvectors.py`](test_inception_leading_eigenvectors.py) | `test_fixture_is_wellformed` | The stored data of Fig. 3D-F (notebook 3) is complete: receptive-field parameters of the V1 and V4 readouts, the frog, and unit-norm eigenvectors of G(x) and J(x). | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 3 draws Fig. 3D-F like the reference picture. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when the eigenvectors are reversed. | yes |
| | `test_recompute_matches_fixture` | Notebook 3 recomputed from the checkpoints gives the stored eigenvectors: the leading 4 of J(x) with \|cos\| > 0.99, the leading one of G(x) with \|cos\| > 0.9 (G(x) is a Monte-Carlo estimate). | no (slow) |
| [`test_resnet50_leading_eigenvectors.py`](test_resnet50_leading_eigenvectors.py) | `test_fixture_is_wellformed` | The stored ResNet-50 eigenvectors of Fig. 4B-C (notebook 3) have the expected shape and unit norm. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 3 draws Fig. 4B-C like the reference picture. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails with other ResNet-50 eigenvectors. | yes |
| | `test_recompute_matches_fixture` | The ResNet-50 eigenvectors recomputed from the checkpoints agree with the stored ones: the leading 4 of J(x) with \|cos\| > 0.99, the leading one of G(x) with \|cos\| > 0.9. | no (slow) |
| [`test_fig4_de_subspace_alignment.py`](test_fig4_de_subspace_alignment.py) | `test_fixture_has_the_four_comparisons_of_both_metrics` | The stored alignment metrics of Fig. 4D-E (notebook 4) have the four comparisons of the figure, for G(x) and J(x). | yes |
| | `test_alignment_is_between_zero_and_one` | Every stored alignment curve has 50 values, all between 0 and 1. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot of notebook 4 draws Fig. 4D-E like the reference picture. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when the curves of one comparison change. | yes |
| | `test_fixture_info_lists_how_the_fixture_was_made` | `fixture_info.json` records the estimator settings, the images and the diffusion checkpoint of each configuration. | yes |
| | `test_fixture_info_agrees_with_the_stored_metrics` | The configurations, images and number of eigenvectors in `fixture_info.json` are those of the stored metrics. | yes |
| | `test_fixture_parameters_match_the_current_settings_of_the_repository` | The fixture was made with the current estimator config and plot defaults, so it still describes this code. | yes |
| [`test_fig5_local_metric_field.py`](test_fig5_local_metric_field.py) | `test_fixture_matrices_are_valid_metrics` | The stored 2 x 2 metrics of Fig. 5 (notebook 5) are symmetric and positive semi-definite, on the grid −2 to 2 around a 64 x 64 image. | yes |
| | `test_fixture_info_lists_how_the_fixture_was_made` | `fixture_info.json` records the settings of each part (samples, grid size, step rule) and the diffusion checkpoints. | yes |
| | `test_fixture_info_agrees_with_the_stored_matrices` | The grid size, metrics and areas in `fixture_info.json` are those of the stored matrices. | yes |
| | `test_fixture_parameters_match_the_current_settings_of_the_repository` | The fixture was made with the current estimator config, job defaults (image, steps) and plot defaults. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot of notebook 5 draws Fig. 5 like the reference picture. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails with another ResNet-50 metric field. | yes |
| [`test_local_metric_field_notebook.py`](test_local_metric_field_notebook.py) | `test_the_notebook_exists_and_has_the_cells_the_tests_use` | Notebook 5 has the code cells that the other tests run. | yes |
| | `test_parameter_cell_gives_the_documented_job_and_its_cost` | The parameter cell of notebook 5 sets the documented job (image 64, a 5 x 5 grid, the step rules of G and J, Inception against ResNet-50) and prints its cost (102 G(x) estimates and 102 J(x) evaluations). | yes |
| | `test_plot_cell_draws_the_reference_figure_from_stored_results` | The plot cell of notebook 5 draws Fig. 5 from stored results like the reference picture. | yes |
| | `test_plot_cell_says_so_when_there_are_no_results` | Without results, the plot cell says "Nothing to plot yet" instead of failing. | yes |
| | `test_notebook_runs_a_small_job_end_to_end` | Notebook 5 runs a small job (J(x) only, V1, a 3 x 3 grid) from launch to plot in a scratch copy of the repository, and gives symmetric metrics. | no (slow) |
| [`test_bures_wasserstein_notebook.py`](test_bures_wasserstein_notebook.py) | `test_the_notebook_has_the_cells_the_tests_use` | Notebook 6 has the code cells that the other tests run. | yes |
| | `test_fixture_info_agrees_with_its_source_and_the_repository` | The stored metrics of Fig. 6 come from the recorded source file (MD5), give the paper's ordering of the comparisons, and match the current estimator config and plot function. | yes |
| | `test_ordering_cell_reports_the_papers_order` | The ordering cell of notebook 6 finds the paper's ordering of the comparisons, for G(x) and J(x). | yes |
| | `test_plot_cell_draws_the_reference_figure` | The plot cell of notebook 6 draws Fig. 6 like the reference picture. | yes |
| | `test_plot_and_ordering_cells_say_so_without_results` | Without results, the cells say "Nothing to check yet" and "Nothing to plot yet" instead of failing. | yes |
| | `test_notebook_runs_stage5_and_plots_in_a_scratch_copy` | Notebook 6 computes the metrics from the per-image results of notebook 4 and plots them, in a scratch copy, without touching the real results. | no (slow) |
| [`test_eigenvectors_per_scale.py`](test_eigenvectors_per_scale.py) | `test_fixture_is_wellformed` | The stored data of notebook 7 (the frog) has 24 increasing noise levels and, for V1 and V4, 4 positive decreasing eigenvalues and unit-norm eigenvectors at each level. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 7 draws the figure of the frog like the reference picture, and saves it. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when the eigenvectors of V1 are reversed. | yes |
| | `test_recompute_matches_fixture` | Notebook 7 recomputed from the checkpoints gives the stored noise levels, and at the coarse levels (τ ≥ 3) the stored leading eigenvector (\|cos\| > 0.9) and leading eigenvalue (within 30%). | no (slow) |
| [`test_stimulus_features_per_scale.py`](test_stimulus_features_per_scale.py) | `test_fixture_is_wellformed` | The stored stimulus features of notebook 8 (the frog) have the noise levels of notebook 7's fixture and 4 positive decreasing eigenvalues and unit-norm eigenvectors at each level. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 8 draws the figure of the frog (stimulus features, then V1 and V4 from notebook 7's fixture) like the reference picture, and saves it. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when the eigenvectors of the stimulus features are reversed. | yes |
| | `test_recompute_matches_fixture` | The stimulus features of the frog recomputed from the V1 checkpoint have the stored settings and noise levels, and at the coarse levels (τ ≥ 3) the stored leading eigenvector (\|cos\| > 0.9) and leading eigenvalue (within 30%). | no (slow) |
| [`test_envelope_spectrum.py`](test_envelope_spectrum.py) | `test_fixture_is_wellformed` | The stored envelope spectrum of notebook 9 has 31 increasing noise levels, a positive curve for each of the 200 test images and both models, the variances of the training images largest first, and a Gaussian bound that decreases from below P. | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 9 draws the envelope spectrum like the reference picture, and saves it. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when the V4 model's curve is made 1.5 times larger. | yes |
| | `test_recompute_matches_fixture` | The spectrum recomputed from the V1 and V4 checkpoints has the stored noise levels and mean curves (within 2%), and the training images give the stored variances and Gaussian bound. | no (slow) |
| [`test_coding_spectrum.py`](test_coding_spectrum.py) | `test_fixture_is_wellformed` | The stored coding spectrum of notebook 10 has 31 increasing noise levels and, for V1 and V4 and each of the 200 test images, the four terms of the curve at each level (positive errors without and with the responses, non-negative changes made by the responses) and a positive trace of J(x). | yes |
| | `test_notebook_plot_reproduces_reference` | The plot cell of notebook 10 draws the coding spectrum like the reference picture, and saves it. | yes |
| | `test_plot_comparison_detects_changes` | The comparison fails when V4's ρ(τ) is made twice larger. | yes |
| | `test_recompute_matches_fixture` | The coding spectrum recomputed from the V1 and V4 checkpoints (about 7 min on a GPU) has the stored noise levels, the stored mean errors without and with the responses (within 2%) and ½∫ρ dτ (within 1 nat, both ways), and the encoders give the stored traces of J(x) (within 0.1%). | no (slow) |
| [`test_getting_started_notebooks.py`](test_getting_started_notebooks.py) | `test_notebook_1_builds_fits_saves_and_reloads_an_encoder` | Notebook 1 runs on a tiny fake TinyImageNet: it builds and fits an encoder of 50 neurons, saves it, and the reloaded encoder predicts the same responses. | yes |
| | `test_notebook_1_checks_the_contract_and_compares_fisher_with_the_teacher` | Notebook 1 checks the encoder against the contract, and compares the leading eigenvectors of its Fisher information with those of the teacher (alignment at k = 1, 5 and 10, between 0 and 1). | yes |
| | `test_notebook_2_trains_a_diffusion_model_and_computes_g` | Notebook 2 trains a small diffusion model and computes finite G(x) and J(x) of the expected shapes. | yes |
| [`test_figures.py`](test_figures.py) | `test_figure_scripts_write_outputs` | Every figure and table script of `figures/` writes its files from a small synthetic `runs/` tree, and skips what is missing (the ON-OFF encoder). | no (slow) |

## Unit tests

14 tests. They check one function on inputs whose answer is known, without stored references, data or checkpoints. They
all run on GitHub Actions, except the 5 that need `compute_g_per_scale`: they are skipped until the `multigeom` that
GitHub Actions installs (the package's `main` branch, or `develop` when testing `develop`) has it.

| File | Test | What it checks | GitHub Actions |
|---|---|---|---|
| [`unit_tests/test_compute_g_per_scale.py`](unit_tests/test_compute_g_per_scale.py) | `test_m_t_is_zero_when_the_denoiser_ignores_the_response` | With a mock denoiser that ignores the response r, both responses give the same noise prediction, so `compute_g_per_scale` returns M_t(x) = 0: every eigenvalue is exactly 0 at every noise level. | skipped until `multigeom` has `compute_g_per_scale` |
| | `test_m_t_is_not_zero_when_the_denoiser_reads_the_response` | The control: the same mock plus a term that reads r gives a positive leading eigenvalue at every noise level, so the zero above is not vacuous. | skipped until `multigeom` has `compute_g_per_scale` |
| | `test_every_difference_is_zero_when_the_responses_are_identical` | With no observation noise (σ = 0) the two responses of each sample are identical, r = r′, so every row of D that the conditional-pair method draws, ε(x_t, t, r) − ε(x_t, t, r′), is exactly zero at every noise level, even with a mock denoiser that reads r. | yes |
| | `test_differences_are_not_zero_when_the_responses_differ` | The control: with the default σ (0.22) the responses differ, and the same mock gives a non-zero row for every sample at every noise level. | yes |
| | `test_trace_of_m_t_is_the_sum_of_squared_differences_over_2nt` | Trace check: at every noise level, the sum of all the eigenvalues of the estimate is tr M̂_t = Σ_s \|δ_s\|² / (2 N t), where δ_s = x̂(x_t, r) − x̂(x_t, r′) is the difference of the two denoised images of sample s (Tweedie's formula, in image units, from the calls recorded by the mock denoiser), N the number of samples and t the noise variance. | skipped until `multigeom` has `compute_g_per_scale` |
| | `test_explicit_matrix_is_symmetric_and_positive_semidefinite` | Explicit matrix check: at every noise level, the whole P x P matrix M̂_t = Σ_s δ_s δ_sᵀ / (2 N t), built from the differences of the denoised images that the estimator drew, is symmetric and positive semi-definite within numerical tolerance (no eigenvalue below −1e-10 times the largest; its zero eigenvalues come out around −1e-16); so is the matrix rebuilt from the eigenpairs that `compute_g_per_scale` returns, and the two agree (relative difference below 1e-5). | skipped until `multigeom` has `compute_g_per_scale` |
| | `test_the_curve_equals_t_times_the_trace_of_the_fisher_information` | The curve trace(M̂_t) equals t · trace(⟨J_t⟩), J_t the Fisher information of the responses about the noisy image, at five noise levels from log t = −6.1 to 8.2 (agreement within 1.3%, tolerance 10%). It uses a linear-Gaussian model whose posterior means and Fisher information are known in closed form, and draws the responses with the spread they have given the noisy image; with the estimator's own draws, given the denoised image, the curve is lower (8% to 99% of it in this model, the coarser the lower), which the test also catches. | skipped until `multigeom` has `compute_g_per_scale` |
| [`unit_tests/test_per_scale_helpers.py`](unit_tests/test_per_scale_helpers.py) | `test_align_signs_flips_only_signs_to_agree_with_coarser_levels` | `align_signs` (`figures/per_scale.py`, notebooks 7 and 8) only flips signs, makes each eigenvector agree with the same one at the next coarser level, and gives the coarsest a positive sum. | yes |
| | `test_compute_summary_on_known_records` | `compute_summary` (`figures/per_scale.py`, notebook 7) gives \|cos\| 0 between images with orthogonal eigenvectors and 1 with the same one, and the fraction of images where V4's leading eigenvalue is larger. | yes |
| | `test_compute_gaussian_env_on_known_variances` | `compute_gaussian_env` (`figures/per_scale.py`, notebook 9), the envelope spectrum of a Gaussian prior, gives Σ_i c_i / (c_i + t) on known variances: the number of directions that vary at tiny noise, ½ per unit-variance direction at t = 1, and tr C / t at huge noise. | yes |
| | `test_compute_information_on_known_curves` | `compute_information` (`figures/per_scale.py`, notebook 10) gives half the area under the curve of each image (trapezoids over log t), the mean over the images and its standard error, on two piecewise-linear curves of known area. | yes |
| | `test_compute_centre_on_known_curves` | `compute_centre` (`figures/per_scale.py`, notebook 10) puts a symmetric bump at its centre, and is not moved by a negative value of the curve. | yes |
| [`test_figures.py`](test_figures.py) | `test_style_keeps_the_notebook_backend_in_a_jupyter_kernel` | Importing `figures.style` in a Jupyter kernel keeps the notebook backend, so figures show as pictures (it switches to Agg only in scripts). | yes |
| | `test_encoder_checkpoint_gives_the_scores_in_both_layouts` | `encoder_checkpoint` reads the test scores of an encoder checkpoint in both layouts (older paper checkpoints and `NeuralEncoder.save`), and returns None when there is none. | yes |
