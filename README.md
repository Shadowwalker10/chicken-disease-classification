# Chicken Disease Classification

## Project Methodology

1. **Update `config.yaml`**  
   Define or update the configuration parameters for the project.

2. **Update `secrets.yaml` [optional]**  
   (Optional) If the project involves sensitive information like API keys or passwords, update them here.

3. **Update `params.yaml`**  
   Set or adjust the parameters for data processing, model training, or evaluation.

4. **Update the entity**  
   Modify or add any entities or data structures used in the project.Don't forget to update constants inside src

5. **Update the configuration manager in `src/config`**  
   Ensure the configuration manager correctly loads and processes the configurations from the YAML files.

6. **Update the components (Data Ingestion, Transformation, Validation)**  
   Modify the data pipeline components to ensure proper data flow and processing.

7. **Update the pipeline**  
   Integrate the components into a cohesive pipeline that handles end-to-end data processing and model training.

8. **Update `main.py`**  
   Adjust the main script to orchestrate the entire process, including calling the pipeline and handling outputs.
