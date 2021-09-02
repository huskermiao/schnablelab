# How to use compressDatasetWorkflow on HCC server
# SD2019-Plant Phenotyping
## Dependencies
* pegasus (does not need to be installed)

## Cloning Project (Need to do only once)
1. You can clone the project using: `git clone https://github.com/freemao/schnablelab.git`
2. Run this command in the location you wish to clone the project to.

## Transferring Data to HCC Server (Need to do only once)
1. This workflow can only be run on the HCC server, so ensure the entire project (schnablelab) is in your work directory on the HCC server.
2. Ensure that the plant dataset is on the HCC server and is accessible by you.
    * Data transfer instructions are below for moving local information to the HCC servers:
        * Mac/Linux (under File Transferring With HCC Supercomputers):<https://hcc.unl.edu/docs/connecting/for_maclinux_users/>
        * Windows (under File Transferring With HCC Supercomputers): <https://hcc.unl.edu/docs/connecting/for_windows_users/>

## Modifying Workflow Paths to Work with File Structure (Need to do only once)
1. Modify file\_paths\_config.py located in the zip folder:
    * Change the data value to reference all of the dataset’s plant folders.
    * Example:
    ` 'data': '/work/csesd/pnnguyen/schnablelab/compressDatasetWorkflow/test_data/*',`
    * data needs to be in this structure for workflow to work for hyperspectral images:
        * data (does not have to be called data)
            * [plant folder name]
            * [plant folder name]
            * ....

## Changing Shell Scripts to Executables (Need to do only once)
1. We will need to change the permissions for our scripts in the pics2predictions folder so we need to run the following commands in the pics2predictions folder:
    * `chmod +x generate_dax.sh` to make generate_dax.sh an executable.
    * `chmod +x plan_dax.sh` to make plan_dax.sh an executable.
    * `chmod +x preprocess.sh` to make preprocess.sh an executable.  

## Running Workflow
1. Run the command:
`./generate_dax.sh [dax file name].dax`
(Where [dax file name] is the name you want for the dax file to generate the dax file
2. Then the command :
`./plan_dax.sh [dax file name].dax`
to plan the dax and run the workflow
3. Use `pegasus-status -l [copy this from the output]` to see the status. (optional)
Example: `pegasus-status -l /work/csesd/pnnguyen/schnablelab/compressDatasetWorkflow/submit/pnnguyen/pegasus/split/run0010`
This is to show the status of the workflow.
3. Use `pegasus-remove [copy this from the output]` to remove the current running workflow. (optional)
Example:`pegasus-remove /work/csesd/pnnguyen/zip/submit/pnnguyen/pegasus/split/run0010`
This is to remove the workflow from the queue.
4. Use `pegasus-analyzer [copy this from the output]` for a failing workflow to see the error message (optional)
Example:`pegasus-analyzer /work/csesd/pnnguyen/schnablelab/compressDatasetWorkflow/submit/pnnguyen/pegasus/split/run0010`
This is to show the error message of the workflow if there was any error.
5. When it’s done, the status should be 100% in the %DONE column (use `pegasus-status` command). And you can see the output in the output folder.
6. The final outputs to the current workflow should be:
  * Each [plant folder name].zip
7. When you want to run the workflow again, make sure to remove the contents in the output and submit folders using `rm -r *` inside those folders.

## Example of using this Workflow
1. In this example, schnablelab and the dataset are all on the HCC sever. Let us go step by step of using this workflow.
2. We are currently in the workflow directory, zip, you can see all of the files of the workflow directory with `ls`. We now need to convert all of the shell scripts into executables. We do this with `chmod +x [file name]` as shown below. The files are now in green after `ls`.
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_chmod.png)
3. Let us check the files that need to be configured to properly run the workflow. Here is the file_paths_config.py file that contains the paths to the dataset's plant folders. Here is shows the path to my test dataset.
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_file_paths_config.png)
4. Now we can generate the dax with `./generate_dax [dax file name].dax` we use test.dax as a test. This now shows test.dax in the directory. We can use plan_dax.sh to plan the dax file and run the workflow. Let us run `./plan_dax [dax file name].dax`.
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_plan_dax.png)
5. We can check the status with the line that was outputted from the plan dax command (`pegasus status -l ...`). The staging in jobs have just begun.
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_status_start.png)
6. After a few minutes with the test dataset, we can see that the workflow is done with (`pegasus status -l ...`). Let us go into our output directory and check the output with `ls`. We can see it outputted several zip files, each one representing a folder in the dataset.
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_output.png)
7. This concludes running the example of running this workflow.

## Workflow Diagram
![](https://github.com/freemao/schnablelab/blob/master/HCCtools/compressDatasetWorkflow/illustrations/zip_workflow.png)
