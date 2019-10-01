# Text Classification using NLP

## Create the python conda env
This will provide you a unique list of python packages needed to run the code
- create a python env based on a list of packages from environment.yml

  ```conda env create -f env/environment.yml```
 - activate the env
  ```conda activate env_nlp_text_class```
 - in case of issue clean all the cache in conda
   ```conda clean -a -y```

## Update or delete the python conda env
- update a python env based on a list of packages from environment.yml
  ```conda env update -f env/environment.yml```
  
- delete the env to recreate it when too many changes are done
  ```conda env remove -n env_nlp_text_class```

## Access of conda env in Jupyter notebook
   To be able to see conda env in Jupyter notebook, you need:
   - the following package in you base env:
   ```conda install nb_conda```

   - the following package in each env (this is the responsibility of the creator of the env to be sure it is in the env)
   ```conda install ipykernel```


