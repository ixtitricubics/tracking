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
