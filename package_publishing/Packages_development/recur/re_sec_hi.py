def second_higest_2(nums):
    if len(nums) < 2:
        return None  # not enough elements

    # Step 1: find largest
    largest = nums[0]
    for x in nums:
        if x > largest:
            largest = x

    # Step 2: find second largest (smaller than largest)
    # Initialize second_largest to something in the list
    second_largest = None

    for x in nums:
        if x != largest:            # skip the largest
            if second_largest is None or x > second_largest:
                second_largest = x

    return second_largest
