# to create a new branch

<pre>
  git checkout -b new_branch_name branch_name_to_copy
  git push -u origin new
</pre>



# to merge two branches
<pre>
git checkout updating_branch_name
git merge branch_name_to_copy_from
</pre>

# if you want the git to store your password
<pre>
git config --global credential.helper store
</pre>
or
<pre>
git config --global credential.helper cache
</pre>
# if you want to change user, then relogin again.
<pre>
gh auth login
</pre>
to show images in docker
<pre>
xhost +si:localuser:root
</pre>
