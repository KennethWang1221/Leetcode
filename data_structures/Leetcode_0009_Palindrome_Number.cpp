#include <iostream>
using namespace std;

class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) 
            return false;
        if (x < 10)
            return true;
        
        int div = 1;
        while (x / 10 >= div) {
            div *= 10;
        }
        
        while (x != 0) {
            int left = x / div;
            int right = x % 10;
            
            if (left != right)
                return false;
                
            x = (x % div) / 10;
            div /= 100;
        }
        
        return true;
    }
};

int main() {
    Solution solution;
    
    cout << "Test case 121: " << solution.isPalindrome(121) << endl; // Expected: 1 (true)
    cout << "Test case -121: " << solution.isPalindrome(-121) << endl; // Expected: 0 (false)
    cout << "Test case 10: " << solution.isPalindrome(10) << endl; // Expected: 0 (false)
    cout << "Test case 0: " << solution.isPalindrome(0) << endl; // Expected: 1 (true)
    cout << "Test case 11: " << solution.isPalindrome(11) << endl; // Expected: 1 (true)
    cout << "Test case 12321: " << solution.isPalindrome(12321) << endl; // Expected: 1 (true)
    cout << "Test case 1001: " << solution.isPalindrome(1001) << endl; // Expected: 1 (true)
    cout << "Test case 1000001: " << solution.isPalindrome(1000001) << endl; // Expected: 1 (true)
    cout << "Test case 12345: " << solution.isPalindrome(12345) << endl; // Expected: 0 (false)
    
    return 0;
}

// g++ -std=c++17 Leetcode_0009_Palindrome_Number.cpp -o test