#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Answer 1


def DeleteSum(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]), dp[i][j - 1] + ord(s2[j - 1]))

    return dp[m][n]


# In[8]:


s1 = "sea"
s2 = "eat"
print(DeleteSum(s1, s2))


# In[10]:


#Answer 2

def ValidString(s):
    stack = []

    for c in s:
        if c == '(' or c == '*':
            stack.append(c)
        elif c == ')':
            if stack and (stack[-1] == '(' or stack[-1] == '*'):
                stack.pop()
            else:
                return False

    while stack and stack[-1] == '*':
        stack.pop()

    return len(stack) == 0


# In[11]:


s = "()"
print(ValidString(s))


# In[12]:


#answer 3

def Steps(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        dp[i][0]= 1
        
    for j in range(1, n + 1):
        dp[j][0] = 1
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j -1]:
                dp[i][j] =dp [i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[m][n]


# In[13]:


word1 = "sea"
word2 = "eat"
print(Steps(word1, word2))


# In[14]:


#anser 4.


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def str2tree(s):
    if not s:
        return None

    stack = []
    i = 0

    while i < len(s):
        if s[i] == '(' or s[i] == ')':
            i += 1
            continue

        j = i
        while j < len(s) and s[j] not in ('(', ')'):
            j += 1

        num = int(s[i:j])
        node = TreeNode(num)

        if stack:
            parent = stack[-1]
            if not parent.left:
                parent.left = node
            else:
                parent.right = node

        stack.append(node)
        i = j

    return stack[-1] if stack else None


# In[18]:


s = "4(2(3)(1))(6(5))"
root = str2tree(s)


# In[24]:


def inorderTraversal(node):
    if not node:
        return[]
    
    result = []
    result.extend(inorderTraversal(node.left))
    result.append(node.val)
    result.extend(inorderTraversal(node.right)) 
    return result


output = inorderTraversal(root)
print("tree is valid")


# In[26]:


#Answer 5

def compress(char):
    i = 0
    j = 0
    count = 1
    
    while i < len(char):
        if i + 1 < len(char) and char[i] == char[ i + 1]:
            count += 1
        else:
            char[j] = char[i]
            j += 1
            if count > 1:
                count_str =str(count)
                for digit in count_str:
                    char[j] = digit
                    j += 1
                count = 1
        i +=1
    return j


# In[32]:


char =  ["a", "a", "b", "b", "c", "c", "c"]
new_lenght = compress(char)
print(new_lenght)


#new_length will be 6, indicating the new length of the compressed array. The first 6 characters of chars will be ["a", "2", "b", "2", "c", "3"].
#Note that the compressed string is modified directly in the input character array chars without using any extra space.


# In[33]:


#Answer 6

from collections import Counter


# In[34]:


def findAnagrams(s, p):
    result = []
    p_freq = Counter(p)
    s_freq = Counter(s[:len(p)])

    if s_freq == p_freq:
        result.append(0)

    for i in range(len(p), len(s)):
        if s_freq[s[i - len(p)]] == 1:
            del s_freq[s[i - len(p)]]
        else:
            s_freq[s[i - len(p)]] -= 1

        s_freq[s[i]] += 1

        if s_freq == p_freq:
            result.append(i - len(p) + 1)

    return result


# In[35]:


s = "cbaebabacd"
p = "abc"
indices = findAnagrams(s, p)
print(indices)


# In[43]:


#Answer 7


def decodeString(s):
    def decodeHelper(s, i ):
        result = ""
        count = 0
        
        while i < len(s):
            if s[i].isdigit():
                count = count * 10 + int(s[i])
            elif s[i] == '[':
                decoded, i = decodeHelper(s, i + 1)
                result += count * decoded
                count = 0
            elif s[i] == ']':
                return result, i
            else:
                result += s[i]
            i += 1

        return result, i

    return decodeHelper(s, 0)[0]                


# In[44]:


s = "3[a]2[bc]"
decoded_string =decodeString(s)
print(decoded_string)


# In[45]:


#Answer 8

def buddyStrings(s, goal):
    if len(s) != len(goal):
        return False

    if s == goal and len(set(s)) < len(s):
        return True

    indices = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            indices.append(i)

    return len(indices) == 2 and s[indices[0]] == goal[indices[1]] and s[indices[1]] == goal[indices[0]]


# In[46]:


s = "ab"
goal = "ba"
print(buddyStrings(s, goal))


# #we can check if there are exactly two indices where the characters in s and goal are different. If there are two such indices, we can swap the characters at those indices in s and check if it becomes equal to goal.

# In[ ]:




