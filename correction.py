import random
import numpy as np
import emip_toolkit as EMTK

# function to generate fixations at each word
def generate_fixations_left_skip_regression(aois_with_tokens):
    
    fixations = []
    word_count = 0
    skip_count = 0
    regress_count = 0
    
    aoi_list = aois_with_tokens.values.tolist()
    
    index = 0
    
    while index < len(aoi_list):
        x, y, width, height, token = aoi_list[index][2], aoi_list[index][3], aoi_list[index][4], aoi_list[index][5], aoi_list[index][7]
        
        word_count += 1
        
        fixation_x = x + width / 3 + random.randint(-10, 10)
        fixation_y = y + height / 2 + random.randint(-10, 10)

        last_skipped = False

        # skipping
        if len(token) < 5 and random.random() < 0.3:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
            last_skipped = True
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
            last_skipped = False
        
        # regressions    
        if  random.random() > 0.96:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            regress_count += 1
        
        index += 1
            
    
    skip_probability = skip_count / word_count
    
    return fixations


# write a function generate offset error as described in the paper "Algorithms for the automated correction of vertical drift in eye-tracking data"
def error_offset(x_offset, y_offset, fixations):
    # x_offset and y_offset are the amount of error in pixels
    # fixations is a list of fixations
    # returns a list of fixations with the offset error added
    # each fixation is a tuple of (x, y, duration)
    # x and y are the coordinates of the fixation
    # the function should return a list of tuples of (x, y, duration)
    offset_error = []
    for fixation in fixations:
        x_error = fixation[0] + x_offset
        y_error = fixation[1] + y_offset
        offset_error.append((x_error, y_error))
    return offset_error

# noise
def error_noise(y_noise_probability, y_noise, duration_noise, fixations):
    '''creates a random error moving a percentage of fixations '''
    
    results = []
    
    for fix in fixations:

        x, y, duration = fix[0], fix[1], fix[2]

        # should be 0.1 for %10
        duration_error = int(duration * duration_noise)

        duration += random.randint(-duration_error, duration_error)

        if duration < 0:
            duration *= -1
        
        if random.random() < y_noise_probability:
            results.append([x, y + random.randint(-y_noise, y_noise), duration])
        else:
            results.append([x, y, fix[2]])
    
    return results


# slope
def error_slope(slope, fixations):
    '''adds a linear slope to the y values of the fixations'''
    results = [] # creates an empty list results to store the results
    
    for fix in fixations: # loop through the list of fixations
        x, y, duration = fix[0], fix[1], fix[2] # unpack the values from each fixation
        y_error = int(x * slope) # calculate the slope error for the y value of the fixation
        results.append([x, y + y_error, duration]) # add the calculated error to the y value and append the result to the `results` list
    
    return results # return the final list of results

    # The slope distortion error is described as a systematic measurement error, meaning that it affects 
    # all fixations in the same way, regardless of the individual fixations. In other words, the slope 
    # distortion error applies to all fixations in the same way, with a fixed magnitude specified by the 
    # parameter dslope. On the other hand, the noise error, within-line regression error, and between-line 
    # regression error are all modeled as random errors, and the magnitude of each 
    # error is determined by a probability parameter. In the case of slope distortion, since the error
    # affects all fixations in the same way, there is no need for a probability parameter.
    
# shift error
def error_shift(shift, fixations, aoi_lines):
    '''adds a constant shift to the y values of the fixations'''

    results = [] # creates an empty list results to store the results

    # Keep track of the line number
    line_number = 0

    # Keep track of the shift error
    y_error = 0
    
    for fix in fixations: # loop through the list of fixations
        x, y, duration = fix[0], fix[1], fix[2] # unpack the values from each fixation

        # If the fixation is on the first line, do not apply the shift distortion error
        if line_number == 0:
            results.append([x, y, duration])
        else:
            results.append([x, y + y_error, duration]) # add the calculated error to the y value and append the result to the `results` list

        # Use the aoi_lines to determine when to increment the line number
        if y > aoi_lines.at[line_number, 'y'] + aoi_lines.at[line_number, 'height']:
            line_number += 1
            # For every pixel that the fixation is below the bottom of the line, increment the shift by shift pixels
            # Does x and y represent pixels?
            y_error += int((y - aoi_lines.at[line_number, 'y']) * shift)

    
    return results # return the final list of results

    # The shift distortion error is described as a systematic measurement error, meaning that it affects 
    # all fixations in the same way, regardless of the individual fixations. In other words, the shift 
    # distortion error applies to all fixations in the same way, with a fixed magnitude specified by the 
    # parameter dshift. On the other hand, the noise error, within-line regression error, and between-line 
    # regression error are all modeled as random errors, and the magnitude of each 
    # error is determined by a probability parameter. In the case of shift distortion, since the error
    # affects all fixations in the same way, there is no need for a probability parameter.

# within line regression
def error_within_line_regression(regression_probability, fixations, words_sorted, aoi_lines, duplicate_fixations = True):
    '''creates a within-line regression error by adding additional fixations to the current line'''
    
    results = [] # creates an empty list `results` to store the results

    # Creating a new fixation list that will be used to store all the additional fixations
    new_fixations = []

    # List to store the current line number
    line_number = 0
    
    for fix in fixations: # loop through the list of fixations
        x, y, duration = fix[0], fix[1], fix[2] # unpack the values from each fixation
        results.append([x, y, duration]) # append the current fixation to the `results` list
        if duplicate_fixations:
            new_fixations.append([x, y, duration])

        # Use the aoi_lines to determine when to increment the line number
        if y > aoi_lines.at[line_number, 'y'] + aoi_lines.at[line_number, 'height']:
            line_number += 1
        
        if random.random() < regression_probability: # if a random number is less than the `regression_probability`, add a regression fixation
            # While loop to get random words until the word is on the current line
            reg_x = 0
            reg_y = 0

            # Getting an array with all the words on the current line
            words_on_line = words_sorted[line_number]

            # Finding the sliced array with words upto the mst recent fixation
            recent_index = 0
            for i in range(1, len(words_on_line)):
                if words_on_line[i][0] > x:
                    recent_index = i
                    break

            if recent_index > 0:

                weights = [1/(i+1) for i in range(recent_index)]

                # Getting a random word from the sliced array such that the most recent words have a higher probability of being selected
                random_word = random.choices(words_on_line[:recent_index], weights=weights, k=1)[0]

                # Getting the x and y coordinates of the random word
                reg_x = random_word[0]
                reg_y = random_word[1]

                # Add the regression fixation to the results
                results.append([reg_x, reg_y, duration])

                if duplicate_fixations:
                    # Adding an extra correct fixation to the fixation list
                    new_fixations.append([x, y, duration])

    if duplicate_fixations:
        return results, new_fixations

    return results # return the final list of results

# between line regression
def error_between_line_regression(regression_probability, fixations, words_sorted, aoi_lines, duplicate_fixations = True):
    '''creates a within-line regression error by adding additional fixations to the current line'''
    
    results = [] # creates an empty list `results` to store the results

    # Creating a new fixation list that will be used to store all the additional fixations
    new_fixations = []

    # List to store the current line number
    line_number = 0
    
    for fix in fixations: # loop through the list of fixations
        x, y, duration = fix[0], fix[1], fix[2] # unpack the values from each fixation
        results.append([x, y, duration]) # append the current fixation to the `results` list
        if duplicate_fixations:
            new_fixations.append([x, y, duration])

        # Use the aoi_lines to determine when to increment the line number
        if y > aoi_lines.at[line_number, 'y'] + aoi_lines.at[line_number, 'height']:
            line_number += 1
        
        if random.random() < regression_probability and line_number > 0: # if a random number is less than the `regression_probability`, add a regression fixation
            # While loop to get random words until the word is on the current line
            reg_x = 0
            reg_y = 0

            # Getting a random line number lesser than the current line number  with more recent lines linearly more probable than less recent lines
            weights = [1 / (i+1) for i in range(line_number)]
            random_line_number = random.choices(range(line_number), weights=weights, k=1)[0] 

            # Getting an array with all the words on the current line
            words_on_line = words_sorted[random_line_number]

            # Getting a random word from the line with no consideration for the most recent fixation or probability
            if len(words_on_line) > 0:
                random_word = random.choices(words_on_line, k=1)[0]

                # Getting the x and y coordinates of the random word
                reg_x = random_word[0]
                reg_y = random_word[1]

                # Add the regression fixation to the results
                results.append([reg_x, reg_y, duration])

                if duplicate_fixations:
                    # Adding an extra correct fixation to the fixation list
                    new_fixations.append([x, y, duration])

    if duplicate_fixations:
        return results, new_fixations
    return results # return the final list of results

# droop

from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
import numpy as np


def draw_fixation(Image_file, fixations):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for fixation in fixations:
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)
        else:
            r = 8
        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        fill_color = (50, 255, 0, 220)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def draw_correction(Image_file, fixations, match_list):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    # Printing the length of the fixations and the match_list
    print('fixations: ', len(fixations))
    print('match_list: ', len(match_list))

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for index, fixation in enumerate(fixations):
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                 r = 5 * (duration / 100)
        else:
            r = 8

        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        if match_list[index] == 1:
        	fill_color = (50, 255, 0, 220)
        else:
        	fill_color = (255, 55, 0, 220)

        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        # text_bound = (x + random.randint(-10, 10), y + random.randint(-10, 10))
        # text_color = (0, 0, 0, 225)
        # font = ImageFont.truetype("arial.ttf", 20)
        # draw.text(text_bound, str(index), fill=text_color,font=font)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def find_lines_Y(aois):
    ''' returns a list of line Ys '''
    
    results = []
    
    for index, row in aois.iterrows():
        y, height = row['y'], row['height']
        
        if y + height / 2 not in results:
            results.append(y + height / 2)
            
    return results



def find_word_centers(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width = row['x'], row['y'], row['height'], row['width']
        
        center = [int(x + width // 2), int(y + height // 2)]
        
        if center not in results:
            results.append(center)
            
    return results


def find_word_centers_and_duration(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width, token = row['x'], row['y'], row['height'], row['width'], row['token']
        
        center = [int(x + width // 2), int(y + height // 2), len(token) * 50]

        if center not in results:
            results.append(center)
    
    #print(results)
    return results



def overlap(fix, AOI):
    """checks if fixation is within AOI"""
    
    box_x = AOI.x
    box_y = AOI.y
    box_w = AOI.width
    box_h = AOI.height

    if fix[0] >= box_x and fix[0] <= box_x + box_w \
    and fix[1] >= box_y and fix[1] <= box_y + box_h:
        return True
    
    else:
        
        return False
    
    
def correction_quality(aois, original_fixations, corrected_fixations):
    
    match = 0
    total_fixations = len(original_fixations)
    results = [0] * total_fixations
    
    for index, fix in enumerate(original_fixations):
        
        for _, row  in aois.iterrows():
            
            if overlap(fix, row) and overlap(corrected_fixations[index], row):
                match += 1
                results[index] = 1
                
    return match / total_fixations, results