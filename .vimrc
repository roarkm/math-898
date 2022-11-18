augroup latex_settings " {
  autocmd FileType python noremap <leader>d <bar> :Silent $(tmux send-keys -t bottom 'make test' C-m)<CR>
  autocmd FileType python noremap <leader>t :wa<CR> <bar> :Silent $(tmux send-keys -t bottom 'python %' C-m)<CR>
augroup END }"
