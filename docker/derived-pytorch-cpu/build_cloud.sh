# Setup env variables
export IMAGE_TAG=dev-v1.1.8
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=custom_container_image_conda
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# Print env variables
echo $PROJECT_ID
echo $IMAGE_URI
echo $IMAGE_REPO_NAME
echo $IMAGE_TAG

# Import config files
cp ../../environment.yml environment.yml
cp ../../deployment/known_hosts known_hosts
cp ../../deployment/id_rsa id_rsa

# Build the image
  
gcloud builds submit --tag $IMAGE_URI . --timeout "2h00m0s"

# Remove the config files
rm environment.yml
rm id_rsa 
rm known_hosts