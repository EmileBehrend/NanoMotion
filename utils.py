import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import mplcursors


def plot_results(pixels, mean, shift_x, shift_y, shift_p, shift_x_y_error, box_shift, fps, res, input_path, output_basepath, plots_dict, rectangles, chop=False,
                 chop_duration=0, start_frame=0):
    print("Started plotting results.")
    opened_plots = []

    position_all = []
    shift_length_all = []
    movement_per_frame_all = []

    output_basename = get_formatted_name(output_basepath)

    for j in range(len(rectangles)):
        my_shift_x = shift_x[j]
        my_shift_y = shift_y[j]
        my_shift_p = shift_p[j]
        my_shift_x_y_error = shift_x_y_error[j]
        my_box_shift = box_shift[j]

        my_shift_x_um = []
        for i in range(len(my_shift_x)):
            my_shift_x_um.append((my_shift_x[i] + my_box_shift[i][0]) * res)

        my_shift_y_um = []
        for i in range(len(my_shift_y)):
            my_shift_y_um.append((my_shift_y[i] + my_box_shift[i][1]) * res)

        my_shift_x_um_error = [e * x for e, x in zip(my_shift_x_y_error, my_shift_x_um)]
        my_shift_y_um_error = [e * y for e, y in zip(my_shift_x_y_error, my_shift_y_um)]

        output_cell_target = "%s_cell_%d" % (output_basepath, j)

        shift_x_step_um = [my_shift_x_um[i + 1] - my_shift_x_um[i]
                           for i in range(len(my_shift_x_um) - 1)]
        shift_y_step_um = [my_shift_y_um[i + 1] - my_shift_y_um[i]
                           for i in range(len(my_shift_y_um) - 1)]

        shift_length_step_um = []
        for i in range(len(shift_x_step_um)):
            shift_length_step_um.append(
                math.sqrt(math.pow(shift_x_step_um[i], 2) + math.pow(shift_y_step_um[i], 2)))

        ls = "--"
        fmt = "o"
        markersize = 4
        if plots_dict["view_position_x"]:
            figure = plt.figure(num=output_cell_target + "x(t), um(s)")
            plt.title("%s\n\nx(t), #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("x, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_x))], my_shift_x_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_x_um_error)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_x(t).png")
            opened_plots.append(figure)

        if plots_dict["view_position_y"]:
            figure = plt.figure(num=output_cell_target + "y(t), um(s)")
            plt.title("%s\n\ny(t), #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("y, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(my_shift_y))], my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize,
                         yerr=my_shift_y_um_error)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_y(t).png")
            opened_plots.append(figure)

        if plots_dict["view_violin"]:
            figure = plt.figure(num=output_cell_target + "violin of step length")
            plt.title("%s\n\nViolin, #%d" % (input_path, j))
            plt.ylabel("Step length, um")

            sns.violinplot(data=shift_length_step_um, inner="stick")

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_violin.png")
            opened_plots.append(figure)

        if plots_dict["view_violin_chop"]:
            number_of_frame_in_a_chop = math.floor(chop_duration * fps)
            number_of_full_chops = math.floor(
                len(shift_length_step_um) / number_of_frame_in_a_chop)

            if number_of_full_chops < 1:
                print("WARNING: chop duration would exceed total number of frames.")
            else:
                figure = plt.figure(num=output_cell_target + "violin chopped of step length")
                plt.title("%s\n\nViolin #%d chopped every %d sec" % (input_path, j, chop_duration))
                plt.xlabel("Frame range")
                plt.ylabel("Step length, um")

                chopped_data = []
                labels = []
                for i in range(number_of_full_chops):
                    chopped_data.append(
                        shift_length_step_um[number_of_frame_in_a_chop * i:number_of_frame_in_a_chop * (i + 1)])

                    labels.append("[%d, %d]" % (start_frame + number_of_frame_in_a_chop *
                                                i, start_frame + number_of_frame_in_a_chop * (i + 1) - 1))

                g = sns.violinplot(data=chopped_data, inner="stick")
                g.set_xticklabels(labels, rotation=30)

                axe = plt.gca()
                axe.legend()
                axe.set_ylim([-0.1, 0.5])

                plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

                plt.savefig(output_cell_target + "_violin_chopped.png")
                opened_plots.append(figure)

        if plots_dict["view_position"]:
            figure = plt.figure(num=output_cell_target + "y(x), um(um)")
            plt.title("%s\n\ny(x), #%d" % (input_path, j))
            plt.xlabel("x, um")
            plt.ylabel("y, um")

            plt.grid()
            # , yerr=my_shift_y_um_error, xerr=my_shift_x_um_error)
            plt.errorbar(my_shift_x_um, my_shift_y_um, ls=ls, fmt=fmt, markersize=markersize)

            axe = plt.gca()
            axe.set_xlim([-8, 8])
            axe.set_ylim([-8, 8])

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_y(x).png")
            opened_plots.append(figure)

        if plots_dict["view_phase"]:
            figure = plt.figure(num=output_cell_target + "phase")
            plt.title("%s\n\nPhase, #%d" % (input_path, j))
            plt.xlabel("Frame #")
            plt.ylabel("Phase")

            plt.grid()
            plt.plot(my_shift_p)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_p.png")
            opened_plots.append(figure)

        if plots_dict["view_step_length"]:
            figure = plt.figure(num=output_cell_target + "steps")
            plt.title("%s\n\nSteps, #%d" % (input_path, j))
            plt.xlabel("t, s")
            plt.ylabel("Length, um")

            plt.grid()
            plt.errorbar([frame / fps for frame in range(len(shift_length_step_um))],
                         shift_length_step_um, ls=None, fmt=fmt, markersize=markersize, alpha=0.5)

            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

            plt.savefig(output_cell_target + "_p.png")
            opened_plots.append(figure)

        if plots_dict["view_position_all_on_one"]:
            position_all.append([my_shift_x_um, my_shift_y_um])

        if plots_dict["view_violin_all_on_one"]:
            shift_length_all.append(shift_length_step_um)

        if plots_dict["view_experimental"]:
            movement_per_frame_all.append(np.sum(shift_length_step_um) / len(shift_x_step_um))

    if plots_dict["view_position_all_on_one"]:
        figure = plt.figure(num=output_basename + "_positions")
        plt.title("%s\n\nAll cells, y(x), #0 to #%d" % (input_path, j))
        plt.xlabel("x, um")
        plt.ylabel("y, um")

        plt.grid()

        bars = []
        for b in range(0, len(position_all)):
            x_raw = position_all[b][0]
            y_raw = position_all[b][1]

            # x -> y
            # y -> -x
            x = [-e for e in y_raw]
            y = x_raw

            bar = plt.errorbar(x, y, ls="-", fmt=fmt, markersize=0,
                               alpha=0.5, label=("Box #%d" % (b)))
            bars.append(bar)

        axe = plt.gca()
        axe.legend()
        # axe.set_xlim([-2, 2])
        # axe.set_ylim([-2, 2])

        mplcursors.cursor(bars, highlight=True)

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

        plt.savefig("%s%s" % (output_basepath, "_positions.png"))
        opened_plots.append(figure)

    if plots_dict["view_violin_all_on_one"]:
        print("Plotting all (%d) violins containing each %d data points." %
              np.shape(shift_length_all))

        figure = plt.figure(num=output_basename + "_violins")
        plt.title("%s\n\nViolins (seaborn), #0 to #%d" % (input_path, j))
        plt.xlabel("Cell #")
        plt.ylabel("Step length, um")

        sns.violinplot(data=shift_length_all, inner="quartiles")

        axe = plt.gca()
        axe.set_ylim([-0.1, 0.5])

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)

        plt.savefig("%s%s" % (output_basepath, "_violins.png"))
        opened_plots.append(figure)

    if plots_dict["view_experimental"]:
        figure = plt.figure(num=output_basename + "_means")
        plt.title("%s\n\nMean movement per frame, #0 to #%d" % (input_path, j))
        plt.xlabel("Cell #")
        plt.ylabel("Length, um")

        ticks = []
        for j in range(0, len(movement_per_frame_all)):
            movement = movement_per_frame_all[j]

            plt.bar(j, movement)
            plt.text(j, movement, "%f" % (movement), ha="center", va="bottom")

            ticks.append(j)

        mean = np.mean(movement_per_frame_all)
        std = np.std(movement_per_frame_all)
        print("Mean movement per frame of %d cells: %f, standard deviation: %f." %
              (len(movement_per_frame_all), mean, std))
        print("%d, %f, %f" % (len(movement_per_frame_all), mean, std))  # raw for easier copy-pasting

        j += 1
        plt.bar(j, mean, yerr=std, capsize=3)
        ticks.append("Mean (%f ± %f)" % (mean, std))

        plt.xticks(range(0, len(ticks)), ticks, rotation=30, ha="right")

        axe = plt.gca()
        axe.set_ylim([0, 0.15])

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15)

        plt.savefig("%s%s" % (output_basepath, "_means.png"))
        opened_plots.append(figure)

        #####

        figure = plt.figure(num=output_basename + "_densities")
        plt.title("%s\n\nDensities, #0 to #%d" % (input_path, j))
        plt.xlabel("Movement, um")
        plt.ylabel("Density")

        for j in range(0, len(shift_length_all)):
            shift_length_step_um = shift_length_all[j]

            sns.distplot(shift_length_step_um, hist=False, kde=True, kde_kws={
                "shade": False, "linewidth": 3})  # , kind="kde")

        # axe = plt.gca()
        # axe.set_ylim([0, 0.15])

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15)

        plt.savefig("%s%s" % (output_basepath, "_densities.png"))
        opened_plots.append(figure)

    plt.show()

    return opened_plots


def export_results(pixels, mean, shift_x, shift_y, shift_p, shift_x_y_error, box_shift, fps, res, output_basepath, rectangles, start_frame=0):
    for j in range(len(rectangles)):
        my_pixels = pixels[j]
        my_mean = mean[j]
        my_shift_x = shift_x[j]
        my_shift_y = shift_y[j]
        my_shift_p = shift_p[j]
        my_shift_x_y_error = shift_x_y_error[j]
        my_box_shift = box_shift[j]

        output_cell_target = "%s_cell_%d.xlsx" % (output_basepath, j)
        print("Exporting results to %s." % (output_cell_target))

        frames = len(my_shift_x)

        df = pd.DataFrame({
            "frame": [frame for frame in range(frames)],
            "t, s": [frame / fps for frame in range(frames)],
            "pixels": my_pixels,
            "mean": my_mean,
            "shift_x, px": my_shift_x,
            "shift_y, px": my_shift_y,
            "xy_error": my_shift_x_y_error,
            "box shift x, px": [shift[0] for shift in my_box_shift],
            "box shift y, px": [shift[1] for shift in my_box_shift],
            "shift_p": my_shift_p
        })

        df = pd.concat([df, pd.DataFrame({
            "fps": [fps],
            "resolution": [res],
            "start_frame": [start_frame],
            "cell_number": [j]
        })], axis=1)

        with pd.ExcelWriter(os.path.join(output_cell_target)) as writer:
            df.to_excel(writer, sheet_name="Sheet 1", index=False)


def get_formatted_name(file):
    if "." in file:
        return os.path.splitext(os.path.basename(file))[0]
    else:
        return os.path.basename(file)


def ensure_directory(file, target_name):
    formatted_name = get_formatted_name(file)

    videofile_dir = os.path.dirname(os.path.abspath(file))
    target_dir = os.path.join(videofile_dir, target_name)
    output_dir = os.path.join(target_dir, formatted_name)

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, formatted_name)
