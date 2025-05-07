#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;

class Solution {
public:
    int carFleet(int target, vector<int>& position, vector<int>& speed) {
        // Create pairs of position and speed, then sort them in reverse order by position
        vector<pair<int, int>> cars;
        for (int i = 0; i < position.size(); i++) {
            cars.push_back({position[i], speed[i]});
        }
        
        // Sort the cars by position in descending order
        sort(cars.rbegin(), cars.rend());
        
        stack<double> stack;  // Stack to track the times it takes for each car to reach the target
        for (const auto& car : cars) {
            double time = (target - car.first) / (double) car.second;  // Calculate time to reach the target
            
            // Check if this car is slower or reaches the target at the same time as the previous car
            if (!stack.empty() && stack.top() >= time) {
                continue;  // This car will be in the same fleet as the previous one
            }

            // Otherwise, this car forms a new fleet
            stack.push(time);
        }
        
        return stack.size();  // The size of the stack represents the number of fleets
    }
};

int main() {
    Solution solution;
    int target = 12;
    vector<int> position = {10, 8, 0, 5, 3};
    vector<int> speed = {2, 4, 1, 1, 3};
    
    int result = solution.carFleet(target, position, speed);
    
    cout << "Number of car fleets: " << result << endl;
    
    return 0;
}


// g++ -std=c++17 Leetcode_0853_Car_Fleet.cpp -o test