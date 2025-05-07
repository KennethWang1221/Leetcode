#include <iostream>
#include <vector>
#include <stack>
using namespace std;

class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        stack<int> stack;  // Stack to simulate the movement of asteroids

        for (int asteroid : asteroids) {
            bool destroyed = false;  // Flag to check if the current asteroid is destroyed

            // Check for collisions with the asteroid at the top of the stack
            while (!stack.empty() && asteroid < 0 && stack.top() > 0) {
                if (stack.top() > -asteroid) {
                    destroyed = true;  // The current asteroid is destroyed
                    break;
                } else if (stack.top() < -asteroid) {
                    stack.pop();  // Pop the asteroid from the stack
                } else {
                    stack.pop();  // Both asteroids are destroyed
                    destroyed = true;
                    break;
                }
            }

            // If the asteroid was not destroyed, push it onto the stack
            if (!destroyed) {
                stack.push(asteroid);
            }
        }

        // Convert the stack to a vector for the final result
        vector<int> result;
        while (!stack.empty()) {
            result.push_back(stack.top());
            stack.pop();
        }

        // Reverse the vector since we pushed elements in reverse order
        reverse(result.begin(), result.end());
        return result;
    }
};

int main() {
    Solution solution;
    
    // Example 1
    vector<int> asteroids1 = {5, 10, -5};
    vector<int> result1 = solution.asteroidCollision(asteroids1);
    cout << "Result 1: ";
    for (int asteroid : result1) {
        cout << asteroid << " ";
    }
    cout << endl;

    // Example 2
    vector<int> asteroids2 = {8, -8};
    vector<int> result2 = solution.asteroidCollision(asteroids2);
    cout << "Result 2: ";
    for (int asteroid : result2) {
        cout << asteroid << " ";
    }
    cout << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0735_Asteroid_Collision.cpp -o test