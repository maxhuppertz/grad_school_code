# Make a simple LaTeX table (only the rows)
def textable(X, fname='table.tex', prec=4):
    # Create formatting string
    fstring = '{:,.'+str(prec)+'f}'

    # Determine shape of input, which is also the shape of the table
    n, k = X.shape

    # Open the file
    with open(fname, mode='w') as file:
        # Go through all rows
        for i in range(n):
            # Start with an empty string for this row
            thisrow = ''

            # Go through all columns
            for j in range(k):
                # Check whether the current entry is a string
                if type(X[i, j]) != str:
                    # If not, check whether it is an integer
                    if type(X[i, j]) != int:
                        # If not, format it, according to the precision
                        # parameter
                        entry = fstring.format(round(X[i, j], 4))
                    else:
                        # If it is an integer, just convert it to a string
                        entry = str(X[i, j])
                else:
                    # If it is, just use it as is
                    entry = X[i, j]

                # Check whether this is the last entry in this row
                if j != k-1:
                    # If not, add an ampersand after the entry
                    delim = r' & '
                else:
                    # If it is, add a double backslash
                    delim = r' \\'

                # Add the entry and delimiter to the row
                thisrow = thisrow + entry + delim

            # Write the row to the file
            file.write(thisrow)
