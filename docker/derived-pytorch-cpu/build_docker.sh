# get the poject ID
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export IMAGE_REPO_NAME=custom_container_image_conda
export IMAGE_TAG=test-1
echo $PROJECT_ID
echo $IMAGE_URI
echo $IMAGE_REPO_NAME
echo $IMAGE_TAG

# checking access
sudo usermod -a -G docker $USER
grep /etc/group -e "docker"
grep /etc/group -e "sudo"

docker build -f Dockerfile -t $IMAGE_URI .
