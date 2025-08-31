def temp_tester(normal_temp):
    def temp_checker(test_temp):
        temp_diff = abs(normal_temp - test_temp)
        if temp_diff <= 1:
            # Healthy temperature range
            return True
        else:
            return False
    return temp_checker

human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)

chicken_tester(42) # True -- i.e. not a fever for a chicken
human_tester(42)   # False -- this would be a severe fever for a human
chicken_tester(43) # False
human_tester(35)   # False -- too low
human_tester(98.6) # False -- normal in degrees F but our reference temp was in degrees C