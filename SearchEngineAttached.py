import search
from PyQt5.QtWidgets import QMainWindow, QPushButton
from SearchEngineUi import Ui_MainWindow
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

class SearchEngineAttached(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        self.show()
        self.widget.hide()  # Hide pagination initially
        self.suggestionBtn.hide()  # Hide suggestion button initially

        self.searchBtn.clicked.connect(self.searchBtnClicked)
        self.searchBox.returnPressed.connect(self.searchBtnClicked)  # Trigger search on Enter key
        self.suggestionBtn.clicked.connect(self.suggestionClicked)

        self.btns = [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5]
        for i, btn in enumerate(self.btns):
            btn.clicked.connect(lambda _, x=i: self.btnClicked(x))

        self.searchResult.setOpenExternalLinks(True)
        self.searchResult.linkActivated.connect(self.openLink)

        self.current_results = []  # Store search results for pagination
        self.current_page = 0  # Track current page
        self.last_suggested_query = None

    def searchBtnClicked(self):
        text = self.searchBox.text().strip()
        if not text:
            return  # Ignore empty search

        # Perform search
        results, elapsed_time = search.search(text) 
        corrected_query = search.correct_spelling(text)  # Runing spell check separately

        # Store results and reset pagination
        self.current_results = results[:50]  # Limit to 50 results
        self.current_page = 0

        # Update UI with first page of results
        self.updateResults()
        self.updatePagination()

        # Update response time
        response_text = f"Response Time: {elapsed_time:.2f} ms"
        self.label.setText(response_text)

        # Handle spelling suggestion
        if corrected_query and corrected_query != text:
            self.last_suggested_query = corrected_query
            self.suggestionBtn.setText(f"Did you mean: {corrected_query}?")
            self.suggestionBtn.show()
        else:
            self.suggestionBtn.hide()

    def suggestionClicked(self):
        if self.last_suggested_query:
            self.searchBox.setText(self.last_suggested_query)
            self.searchBtnClicked()

    # Displays search results for the current page
    def updateResults(self):
        start_index = self.current_page * 10
        end_index = start_index + 10
        search_result_html = ""

        if not self.current_results:
            search_result_html = "<p><i>No results found.</i></p>"
        else:
            for doc, _ in self.current_results[start_index:end_index]:
                search_result_html += f'<a href="{doc}">{doc}</a><br><br>'

        self.searchResult.setText(search_result_html)
        self.widget.setVisible(bool(self.current_results))  # Showing pagination only if there are results

    # Updates pagination buttons based on results count
    def updatePagination(self):
        total_pages = min(5, -(-len(self.current_results) // 10))  # Ceiling division

        for i, btn in enumerate(self.btns):
            btn.setVisible(i < total_pages)  # Show only necessary buttons
            btn.setStyleSheet(self.defaultBtnStyle())

        if total_pages > 0:
            self.btns[self.current_page].setStyleSheet(self.activeBtnStyle())  # Highlight current page

    # Handles pagination button clicks
    def btnClicked(self, btn_index):
        self.current_page = btn_index
        self.updateResults()
        self.updatePagination()

    def openLink(self, url):
        QDesktopServices.openUrl(QUrl(url))

    def defaultBtnStyle(self):
        """ Default pagination button style. """
        return ("font-size: 18pt; border-style: solid; border-width: 1px; "
                "border-color: rgb(82, 122, 175); border-radius: 5px; "
                "background-color: rgb(235, 255, 255); color: rgb(82, 122, 175);")

    def activeBtnStyle(self):
        """ Active pagination button style. """
        return ("font-size: 18pt; border-style: solid; border-width: 1px; "
                "border-color: rgb(82, 122, 175); border-radius: 5px; "
                "background-color: lightblue; color: rgb(82, 122, 175);")
