import docker
import logging
from pathlib import Path
import sys

class DockerBuilder:
    def __init__(self, dockerfile_path: str, tag: str):
        """
        Initialize the Docker builder.
        
        Args:
            dockerfile_path (str): Path to the Dockerfile
            tag (str): Tag for the Docker image
        """
        self.dockerfile_path = Path(dockerfile_path)
        self.tag = tag
        self.client = docker.from_env()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_dockerfile(self) -> bool:
        """
        Validate if Dockerfile exists and is accessible.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not self.dockerfile_path.exists():
            self.logger.error(f"Dockerfile not found at {self.dockerfile_path}")
            return False
        
        if not self.dockerfile_path.is_file():
            self.logger.error(f"{self.dockerfile_path} is not a file")
            return False
            
        return True
    
    def build_image(self) -> tuple:
        """
        Build Docker image from Dockerfile.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        if not self.validate_dockerfile():
            return False, "Dockerfile validation failed"
        
        try:
            self.logger.info(f"Building Docker image with tag: {self.tag}")
            
            # Get the build context (directory containing Dockerfile)
            build_context = str(self.dockerfile_path.parent)
            
            # Build the Docker image
            image, build_logs = self.client.images.build(
                path=build_context,
                dockerfile=str(self.dockerfile_path.name),
                tag=self.tag,
                rm=True  # Remove intermediate containers
            )
            
            # Log the build output
            for log in build_logs:
                if 'stream' in log:
                    self.logger.info(log['stream'].strip())
            
            self.logger.info(f"Successfully built image: {self.tag}")
            return True, f"Image built successfully: {self.tag}"
            
        except docker.errors.BuildError as e:
            self.logger.error(f"Build error: {str(e)}")
            return False, f"Build failed: {str(e)}"
        except docker.errors.APIError as e:
            self.logger.error(f"Docker API error: {str(e)}")
            return False, f"API error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return False, f"Unexpected error: {str(e)}"

def main():
    """
    Main function to demonstrate usage of DockerBuilder.
    """
    # if len(sys.argv) != 3:
    #     print("Usage: python docker_builder.py <dockerfile_path> <image_tag>")
    #     sys.exit(1)
    
    dockerfile_path = sys.argv[1]
    image_tag = sys.argv[2]
    # print(dockerfile_path, image_tag)
    # exit(0)
    
    builder = DockerBuilder(dockerfile_path, image_tag)
    success, message = builder.build_image()
    
    if not success:
        sys.exit(1)
    
if __name__ == "__main__":
    main()