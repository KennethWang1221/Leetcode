#include <iostream>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution{    
    public:    
        vector<int> getConcatenation(vector<int>& nums){            
            vector<int> ans;            
            int len;            
            len = nums.size();            
            for(int i = 0; i < 2 * len; i++){                
                ans.push_back(nums[i % len]);                
            }
            return ans;            
        }
};


int main() {
    Solution sol;
    vector<int> input = {1,2,1};
    vector<int> result = sol.getConcatenation(input);
    cout << "Test case1:";
    for (int num: result){
        cout << num << " ";
    }
    cout << endl;
    return 0;
}

// g++ -std=c++17 Leetcode_1929_Concatenation_of_Array.cpp -o test