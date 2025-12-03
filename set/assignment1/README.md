# CS4015 Software engineering and testing for AI systems

This folder contains the example for converting your model to onnx, using the onnx runtime.

## Model Training: Good vs Bad Models

This notebook (`models.ipynb`) demonstrates two approaches to training AdaBoost classifiers: a baseline model without fairness considerations (BAD model) and a fairness-aware model using post-processing techniques (GOOD model).

### Dataset

The models are trained on synthetic data (`data/synth_data_for_training.csv`) with the following setup:
- **Target variable**: `checked` (binary classification)
- **Sensitive features**: 
  - `persoon_geslacht_vrouw` (gender)
  - `pla_hist_pla_categorie_doelstelling_16` (category)
- **Train/Test split**: 75% training, 25% testing (random_state=42)

### BAD Model: Baseline Without Fairness Considerations

**Approach**: Standard AdaBoost classifier with no preprocessing or post-processing for fairness.

- Achieves high overall accuracy (~94.88%)
- **Exhibits bias**: Significant accuracy differences between groups
   - Gender groups: ~1.2% difference
   - Category groups: ~6% difference

### GOOD Model: Fairness-Aware Post-Processing

**Approach**: AdaBoost classifier with post-processing using group-specific thresholds to equalize accuracy across sensitive groups.

**Training Process**:

1. **Initial Model Training**:
   - Split training data: 70% for model training, 30% for validation (threshold finding)
   - AdaBoostClassifier with 100 estimators, learning_rate=1.0
   - Base estimator: DecisionTreeClassifier with max_depth=2
   - Feature selection: VarianceThreshold preprocessing step
   - Trained on `X_train_model` (70% of training data)

2. **Threshold Optimization** (Post-processing):
   - **Goal**: Find group-specific thresholds `t₀` and `t₁` that equalize accuracy between groups
   - **Process**:
     - Get probability scores from the trained model on validation set
     - For each sensitive feature, split validation data by group (0.0 vs 1.0)
     - **Two-stage optimization**:
       - **Stage 1**: Coarse grid search (200×200 threshold combinations)
       - **Stage 2**: Fine refinement (500×500 combinations) around best candidates
     - Optimize to minimize absolute accuracy difference: `|acc₀ - acc₁|`
   
3. **Final Model Training**:
   - Retrain AdaBoost on the full training set (`X_train`) after thresholds are found
   - Same architecture as initial model

4. **Prediction**:
   - Get probability scores from the model
   - Apply group-specific thresholds based on sensitive feature values
   - If a sample matches multiple sensitive groups, use the first matching threshold
   - Default threshold (0.5) for samples not matching any group

**Key Features**:
- **Fairness without model modification**: Thresholds are applied at inference time, not during training
- **Equalized accuracy**: Reduces accuracy gaps between groups
- **Maintains performance**: Overall accuracy remains competitive while improving fairness

**Advantages**:
- Model internals remain unchanged (fairness achieved through post-processing)
- Can be applied to any trained classifier
- Transparent and interpretable (thresholds can be inspected)
- No need to modify training data or remove features

### Model Comparison

| Aspect | BAD Model | GOOD Model |
|--------|-----------|------------|
| **Preprocessing** | None | VarianceThreshold |
| **Fairness Method** | None | Post-processing (threshold per group) |
| **Overall Accuracy** | ~94.88% | ~95.07% |
| **Bias Reduction** | No | Yes (reduced accuracy gaps) |
| **Model Complexity** | Simple pipeline | Same + threshold optimization |

### ONNX Conversion

Both models are converted to ONNX format:
- **BAD model**: `model/bad_randomforest.onnx` (note: despite filename, uses AdaBoost)
- **GOOD model**: `model/good_adaboost.onnx`

**Important Note**: The ONNX models use the standard 0.5 threshold. For the GOOD model, group-specific thresholds must be applied at inference time outside the ONNX model.

### Usage

1. **Install dependencies**:
   ```bash
   pipenv shell
   pipenv install
   ```

2. **Run the notebook**:
   - Execute cells sequentially in `models.ipynb`
   - Models will be trained and saved to `model/` directory

3. **Load and use models**:
   ```python
   import onnxruntime as rt
   
   # Load BAD model
   bad_session = rt.InferenceSession("model/bad_randomforest.onnx")
   predictions = bad_session.run(None, {'X': X_test.values.astype(np.float32)})
   
   # Load GOOD model (apply thresholds separately)
   good_session = rt.InferenceSession("model/good_adaboost.onnx")
   scores = good_session.run(None, {'X': X_test.values.astype(np.float32)})
   # Apply group-specific thresholds (see predict_with_group_thresholds function)
   ```

---

### <u>Pipenv</u>

To make life easy for everyone, we've setup a pip file to ensure quick and easy install of dependencies.

<b>NOTE</b>: <i>Before installing any dependencies, open the Pipfile in your editor and un-comment the version of Tensorflow for your system.</i>

Open a terminal/ powershell and navigate to the project folder, then type:

    pipenv shell

This will put your current session into the python virtual environment. Then type:

    pipenv install

This will install the dependencies defined in the Pipfile, into this specific environment. By doing this, we can ensure no cross dependency issues when working on different python projects.

---

You will need to enable this virtual environment in your code editor to ensure it uses the correct dependencies. For VS Code, this can be found in the bottom right corner of the UI.

It will currently likely show your current Python version. Click this and it will open up the 'Select Interpreter' drop down. For myself, the environment starts with <b><i>'labs'</i></b>, which I then click on to enable as my interpreter.

Yours will likely be the same, or if different, will be shown in your terminal/ powershell window when you typed 'pipenv shell' before.

That should have you up and running! Enjoy the labs and if you have any issues with this, please reach out to the staff and we'll do our best to get you going.
