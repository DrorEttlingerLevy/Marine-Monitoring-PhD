_Jekyll_ is software that helps you “generate” or create a _static website_
![programming historian web](https://programminghistorian.org/en/lessons/building-static-sites-with-jekyll-github-pages)

for static website there is no need of database and it is easy to maintenance

Jekyll is not officially supported by Windows

The default command line program is called “Terminal” on Macs (located in _Applications > Utilities_), and “Command Prompt”, “Windows Power Shell”, or “Git Bash” on Windows (these are three different options that each differ in the type of commands they accept)
[[LinkedIn posts]]

1. Jekyll is built from the [Ruby coding language](https://en.wikipedia.org/wiki/Ruby_%28programming_language%29), and we’ll need to install it in your environment. Visit [https://rubyinstaller.org/downloads/](https://rubyinstaller.org/downloads/) and download the latest version of the installer with the DevKit. (The website suggests the most appropriate installer for you, just look for the `=>` symbol before each link.)
2. the installer will prompt a message asking if you want to run **ridk install**. Keep this option selected and click on “Finish”.
3. After closing the installer, a new command prompt will ask which components you want to install. Type ‘3’ (no quotes). It will install **MSYS2 and MINGW development toolchain**.
following message -> Install MSYS2 and MINGW development toolchain succeeded
install jekyll -> `gem install jekyll bundler`
`jekyll -v`

cd to the git desktop path -> 
`gem install jekyll bundler`
`jekyll new Site`
`cd Site`
![[Pasted image 20241203062147.png]]

### see the web locally
`bundle exec jekyll serve --watch`
*jekyll serve* tells your computer to run Jekyll locally. to stop this run ctrl-c.

changes in `_config.yml` will not show without restarting jekyll

localhost:4000 -> to see local web
later --> localhost:4000/site/

### web setting via _config.yml
added a new `“_site”` folder. This is where Jekyll puts the HTML files it generates from the other files in your website folder. Jekyll works by taking various files like your site configuration settings (`_config.yml`) and files that just contain post or page content without other webpage information (e.g. about.md), putting these all together, and spitting out HTML pages that a web browser is able to read and display to site visitors.

- **baseurl**: Fill in the quotation marks with a forward slash followed by the name of your website folder (e.g. “/JekyllDemo/”) to help locate the site at the correct URL. Make sure that your folder is the same the GitHub repository name and ends with a backslash (`/`). It will be required for publishing it on GitHub Pages.
- **url**: Replace “http://yourdomain.com” with “localhost:4000” to help locate your local version of the site at the correct URL.
The changes you made to the _baseurl_ and _url_ lines will let your site run from the same files both locally on your computer and live on the Web, but **doing this changed the URL where you’ll see your local site from now on** (while [Jekyll is running](https://programminghistorian.org/en/lessons/building-static-sites-with-jekyll-github-pages#section3-1)) from localhost:4000 to **localhost:4000/JekyllDemo/** (substitute your website folder name for _JekyllDemo_ and remembering the last slash mark).

![[Pasted image 20241206094927.png]]




