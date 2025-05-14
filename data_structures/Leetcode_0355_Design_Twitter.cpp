#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
using namespace std;

class Twitter {
private:
    struct Tweet {
        int id, time;
        Tweet(int i, int t) : id(i), time(t) {}
    };

    unordered_map<int, unordered_set<int>> followers; // userId -> followers
    unordered_map<int, vector<Tweet>> tweets; // userId -> list of tweets
    int timestamp;

public:
    Twitter() : timestamp(0) {}

    // Post a new tweet
    void postTweet(int userId, int tweetId) {
        tweets[userId].emplace_back(tweetId, timestamp++);
    }

    // Get the news feed for a user
    vector<int> getNewsFeed(int userId) {
        vector<int> res;
        priority_queue<pair<int, int>> pq;  // Pair of (time, tweetId)

        unordered_set<int> followeeList = followers[userId];
        followeeList.insert(userId);  // Include self-tweets

        // Collect tweets from all followed users
        for (int followee : followeeList) {
            auto &tw = tweets[followee];
            for (int i = max(0, (int)tw.size() - 10); i < tw.size(); ++i) {
                pq.emplace(tw[i].time, tw[i].id);  // Use time to get the most recent
            }
        }

        // Extract the most recent tweets (up to 10)
        while (!pq.empty() && res.size() < 10) {
            res.push_back(pq.top().second);  // Push tweetId
            pq.pop();
        }

        return res;
    }

    // Follow another user
    void follow(int followerId, int followeeId) {
        if (followerId != followeeId) {
            followers[followerId].insert(followeeId);
        }
    }

    // Unfollow a user
    void unfollow(int followerId, int followeeId) {
        followers[followerId].erase(followeeId);
    }
};

int main() {
    Twitter twitter;

    // Test sequence
    twitter.postTweet(1, 5);  // User 1 posts a tweet (id = 5)
    vector<int> newsFeed1 = twitter.getNewsFeed(1);  // Should return [5]
    cout << "[null, null, ";
    for (int tweet : newsFeed1) {
        cout << tweet << " ";
    }
    cout << "]" << endl;

    twitter.follow(1, 2);  // User 1 follows user 2
    twitter.postTweet(2, 6);  // User 2 posts a tweet (id = 6)
    vector<int> newsFeed2 = twitter.getNewsFeed(1);  // Should return [6, 5]
    cout << "[null, null, ";
    for (int tweet : newsFeed2) {
        cout << tweet << " ";
    }
    cout << "]" << endl;

    twitter.unfollow(1, 2);  // User 1 unfollows user 2
    vector<int> newsFeed3 = twitter.getNewsFeed(1);  // Should return [5] as user 1 is no longer following user 2
    cout << "[null, null, null, ";
    for (int tweet : newsFeed3) {
        cout << tweet << " ";
    }
    cout << "]" << endl;

    return 0;
}

// g++ -std=c++17 Leetcode_0355_Design_Twitter.cpp -o test      