import gymnasium as gym
import pygame
import numpy as np

class ObjectDetectionEnv(gym.Env):
    def __init__(self, grid_size=10, num_objects=3, detection_radius=1):
        super(ObjectDetectionEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # Four actions: Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)

        # Define environment parameters
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.detection_radius = detection_radius
        self.objects = np.zeros((self.num_objects, 2), dtype=int)  # Object positions (x, y)
        self.agent_pos = np.array([0, 0])  # Agent's initial position
        self.current_step = 0

        # Initialize objects' positions randomly within the grid
        self._reset_objects()

    def reset(self):
        # Reset the environment to its initial state
        self.agent_pos = np.array([0, 0])
        self.current_step = 0
        self._reset_objects()
        return self._get_observation()

    def step(self, action):
        # Take an action (0: Up, 1: Down, 2: Left, 3: Right)
        self._move_agent(action)
        detected_objects = self._detect_objects()

        reward = len(detected_objects)  # Reward based on the number of detected objects
        self.current_step += 1
        done = self.current_step >= 50  # End episode after 50 steps

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        # Render the environment using Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((255, 255, 255))  # Clear the screen

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(screen, (0, 0, 0), (i * 30, j * 30, 30, 30), 1)

        # Draw agent
        pygame.draw.rect(screen, (0, 0, 255), (self.agent_pos[1] * 30, self.agent_pos[0] * 30, 30, 30))

        # Draw objects
        for obj in self.objects:
            pygame.draw.rect(screen, (255, 0, 0), (obj[1] * 30, obj[0] * 30, 30, 30))

        pygame.display.flip()  # Update the display

    def _reset_objects(self):
        # Randomly initialize objects' positions within the grid
        self.objects = np.random.randint(0, self.grid_size, size=(self.num_objects, 2))

    def _move_agent(self, action):
        # Move the agent based on the chosen action
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

    def _detect_objects(self):
        # Detect objects within the agent's detection radius
        detected_objects = []
        for i, obj in enumerate(self.objects):
            distance = np.linalg.norm(obj - self.agent_pos)
            if distance <= self.detection_radius:
                detected_objects.append(i)
        return detected_objects

    def _get_observation(self):
        # Generate an observation representing the grid
        observation = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        observation[self.agent_pos[0]][self.agent_pos[1]] = 1.0  # Agent's position
        for obj in self.objects:
            observation[obj[0]][obj[1]] = 0.5  # Object positions
        return observation

# Initialize Pygame
pygame.init()
screen_size = (300, 300)
screen = pygame.display.set_mode(screen_size)

# Example usage:
if __name__ == "__main__":
    env = ObjectDetectionEnv()
    obs = env.reset()
    total_reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        pygame.time.delay(200)  # Delay for smoother visualization

        if done:
            break

    total_objects = env.num_objects  # Get the total number of objects
    print("Total Reward:", total_reward)
    print("Total Objects:", total_objects)

    pygame.quit()  # Close the Pygame window when done
