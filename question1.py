def find_pattern(A, B):             # A is the main list and B is the pattern list
    occurrences = []                # List to store the indices of the pattern occurrences
    lenA = len(A)                   # Length of the main list A
    lenB = len(B)                   # Length of the pattern list B
    
    for i in range(lenA - lenB + 1):    # Iterate through the main list A
        match = True                    # Assume that the pattern is found
        for j in range(lenB):           # Iterate through the pattern list B
            if A[i + j] != B[j]:         
                match = False           # If the elements do not match set match to false, break the inner loop
                break                   
        if match:                    
            occurrences.append(i)   # If the pattern is found append the index to the occurrences list
            i += lenB - 1           # Skip the pattern length to avoid overlapping
    
    return occurrences              # Return the list of occurrences

# Test the function
A = [1, 70, 9, 1, 2, 30, 6, 1, 2, 30, 50]
B = [1, 2, 30]
print(find_pattern(A, B))



