#include <iostream>
#include <vector>
#include <algorithm>  // for sort
using namespace std;

class Solution {
public:
    int numRescueBoats(vector<int>& people, int limit) {
        int n = people.size();
        int l = 0, r = n - 1;
        int res = 0;
        
        // Sort the people array
        sort(people.begin(), people.end());

        while (l <= r) {
            int total = people[l] + people[r];
            
            if (total <= limit) {
                res++;  // Pair the lightest and heaviest person together
                l++;
                r--;
            } else {
                res++;  // Heaviest person goes alone
                r--;
            }
        }

        return res;
    }
};

int main() {
    Solution solution;
    vector<int> people = {3, 5, 3, 4};
    int limit = 5;

    int result = solution.numRescueBoats(people, limit);
    cout << "Minimum number of boats required: " << result << endl;

    return 0;
}
// g++ -std=c++17 Leetcode_0881_Boats_to_Save_People.cpp -o test